#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "echoedge_v6_model.h"


/* ============================================================
 * Hardware
 * ============================================================ */

#define I2S_BCLK GPIO_NUM_4
#define I2S_WS   GPIO_NUM_5
#define I2S_SD   GPIO_NUM_6
#define LED_GPIO GPIO_NUM_7


/* ============================================================
 * V6 training / preprocessing parameters
 * ============================================================ */

#define SAMPLE_RATE       16000
#define WINDOW_SAMPLES    16000
#define HOP_SAMPLES       8000

#define N_FFT             400
#define HOP_LENGTH        160

#define N_MELS            40
#define N_FRAMES          101


/* ============================================================
 * V6 INT8 quantization
 * ============================================================ */

#define INPUT_SCALE       0.035467877984046936f
#define INPUT_ZERO_POINT  (-26)

#define OUTPUT_SCALE      0.00390625f
#define OUTPUT_ZERO_POINT (-128)


/* ============================================================
 * Detection
 * ============================================================ */

#define WAKE_THRESHOLD    0.60f
#define LED_HOLD_MS       2000

#define TAG "ECHOEDGE"


/* ============================================================
 * I2S
 * ============================================================ */

static i2s_chan_handle_t rx_handle = NULL;


/* ============================================================
 * Feature buffers
 * ============================================================ */

static int16_t *audio_window = NULL;

static float *mel_db = NULL;

static float *feature = NULL;

static int8_t *input_quant = NULL;


/* ============================================================
 * TFLite Micro
 * ============================================================ */

static tflite::MicroInterpreter *interpreter = NULL;

static TfLiteTensor *input_tensor = NULL;

static TfLiteTensor *output_tensor = NULL;


/* ============================================================
 * Bluestein FFT
 *
 * Exact DFT length:
 *
 *      N = 400
 *
 * Internal convolution FFT:
 *
 *      M = 1024
 *
 * This preserves the 400-point frequency grid used by V6.
 * ============================================================ */

#define BLU_N         400
#define BLU_FFT_SIZE  1024


/* Chirp:
 *
 * exp(+j*pi*n^2/N)
 */

static float blu_chirp_re[BLU_N];

static float blu_chirp_im[BLU_N];


/* FFT of fixed convolution kernel */

static float blu_kernel_fft_re[BLU_FFT_SIZE];

static float blu_kernel_fft_im[BLU_FFT_SIZE];


/* Working FFT buffers */

static float blu_work_re[BLU_FFT_SIZE];

static float blu_work_im[BLU_FFT_SIZE];


static bool blu_fft_ready = false;


/* ============================================================
 * Radix-2 complex FFT
 * ============================================================ */

static void radix2_fft(
    float *re,
    float *im,
    int n,
    bool inverse)
{
    /* --------------------------------------------------------
     * Bit reversal
     * -------------------------------------------------------- */

    for (int i = 1, j = 0;
         i < n;
         ++i) {

        int bit = n >> 1;

        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }

        j ^= bit;

        if (i < j) {

            float tr = re[i];
            re[i] = re[j];
            re[j] = tr;

            float ti = im[i];
            im[i] = im[j];
            im[j] = ti;
        }
    }


    /* --------------------------------------------------------
     * Cooley-Tukey stages
     * -------------------------------------------------------- */

    for (int len = 2;
         len <= n;
         len <<= 1) {

        const int half =
            len >> 1;

        const float angle =
            (inverse ? 1.0f : -1.0f) *
            2.0f *
            (float)M_PI /
            (float)len;

        const float wlen_re =
            cosf(angle);

        const float wlen_im =
            sinf(angle);


        for (int base = 0;
             base < n;
             base += len) {

            float w_re = 1.0f;
            float w_im = 0.0f;


            for (int j = 0;
                 j < half;
                 ++j) {

                const int u =
                    base + j;

                const int v =
                    u + half;


                const float v_re =
                    re[v] * w_re -
                    im[v] * w_im;

                const float v_im =
                    re[v] * w_im +
                    im[v] * w_re;


                const float u_re =
                    re[u];

                const float u_im =
                    im[u];


                re[u] =
                    u_re + v_re;

                im[u] =
                    u_im + v_im;

                re[v] =
                    u_re - v_re;

                im[v] =
                    u_im - v_im;


                const float next_w_re =
                    w_re * wlen_re -
                    w_im * wlen_im;

                const float next_w_im =
                    w_re * wlen_im +
                    w_im * wlen_re;


                w_re =
                    next_w_re;

                w_im =
                    next_w_im;
            }
        }
    }


    /* --------------------------------------------------------
     * Inverse normalization
     * -------------------------------------------------------- */

    if (inverse) {

        const float inv_n =
            1.0f / (float)n;

        for (int i = 0;
             i < n;
             ++i) {

            re[i] *= inv_n;
            im[i] *= inv_n;
        }
    }
}


/* ============================================================
 * Initialize exact 400-point Bluestein FFT
 * ============================================================ */

static bool init_400_fft_tables(void)
{
    const int N =
        BLU_N;

    const int M =
        BLU_FFT_SIZE;


    /* --------------------------------------------------------
     * Chirp:
     *
     * c[n] = exp(j*pi*n^2/N)
     * -------------------------------------------------------- */

    for (int n = 0;
         n < N;
         ++n) {

        const float angle =
            (float)M_PI *
            (float)n *
            (float)n /
            (float)N;

        blu_chirp_re[n] =
            cosf(angle);

        blu_chirp_im[n] =
            sinf(angle);
    }


    /* --------------------------------------------------------
     * Clear global work buffers.
     *
     * Do NOT allocate large local arrays here.
     * -------------------------------------------------------- */

    memset(
        blu_work_re,
        0,
        sizeof(blu_work_re));

    memset(
        blu_work_im,
        0,
        sizeof(blu_work_im));


    /* --------------------------------------------------------
     * Build Bluestein convolution kernel
     *
     * b[m] = exp(j*pi*m^2/N)
     *
     * for positive and wrapped negative indices.
     * -------------------------------------------------------- */

    blu_work_re[0] =
        1.0f;

    blu_work_im[0] =
        0.0f;


    for (int m = 1;
         m < N;
         ++m) {

        /* Positive m */

        blu_work_re[m] =
            blu_chirp_re[m];

        blu_work_im[m] =
            blu_chirp_im[m];


        /* Wrapped negative m */

        blu_work_re[M - m] =
            blu_chirp_re[m];

        blu_work_im[M - m] =
            blu_chirp_im[m];
    }


    /* --------------------------------------------------------
     * FFT the fixed kernel
     * -------------------------------------------------------- */

    radix2_fft(
        blu_work_re,
        blu_work_im,
        M,
        false);


    /* --------------------------------------------------------
     * Save transformed kernel
     * -------------------------------------------------------- */

    memcpy(
        blu_kernel_fft_re,
        blu_work_re,
        sizeof(blu_kernel_fft_re));

    memcpy(
        blu_kernel_fft_im,
        blu_work_im,
        sizeof(blu_kernel_fft_im));


    blu_fft_ready =
        true;


    ESP_LOGI(
        TAG,
        "400-point Bluestein FFT initialized "
        "(internal FFT size=%d)",
        M);


    return true;
}


/* ============================================================
 * Compute exact 400-point DFT power spectrum
 * ============================================================ */

static void compute_power_spectrum_400(
    const float *samples,
    float *power_out)
{
    const int N =
        BLU_N;

    const int M =
        BLU_FFT_SIZE;


    if (!blu_fft_ready) {

        memset(
            power_out,
            0,
            (N / 2 + 1) *
            sizeof(float));

        return;
    }


    /* --------------------------------------------------------
     * a[n] =
     *
     * x[n] * exp(-j*pi*n^2/N)
     *
     * -------------------------------------------------------- */

    memset(
        blu_work_re,
        0,
        sizeof(blu_work_re));

    memset(
        blu_work_im,
        0,
        sizeof(blu_work_im));


    for (int n = 0;
         n < N;
         ++n) {

        const float x =
            samples[n];


        blu_work_re[n] =
            x *
            blu_chirp_re[n];

        blu_work_im[n] =
            -x *
            blu_chirp_im[n];
    }


    /* --------------------------------------------------------
     * FFT(a)
     * -------------------------------------------------------- */

    radix2_fft(
        blu_work_re,
        blu_work_im,
        M,
        false);


    /* --------------------------------------------------------
     * FFT(a) * FFT(b)
     * -------------------------------------------------------- */

    for (int k = 0;
         k < M;
         ++k) {

        const float ar =
            blu_work_re[k];

        const float ai =
            blu_work_im[k];

        const float br =
            blu_kernel_fft_re[k];

        const float bi =
            blu_kernel_fft_im[k];


        blu_work_re[k] =
            ar * br -
            ai * bi;

        blu_work_im[k] =
            ar * bi +
            ai * br;
    }


    /* --------------------------------------------------------
     * Inverse FFT
     * -------------------------------------------------------- */

    radix2_fft(
        blu_work_re,
        blu_work_im,
        M,
        true);


    /* --------------------------------------------------------
     * Recover X[k]
     *
     * X[k] =
     * convolution[k] *
     * exp(-j*pi*k^2/N)
     * -------------------------------------------------------- */

    for (int k = 0;
         k <= N / 2;
         ++k) {

        const float cr =
            blu_work_re[k];

        const float ci =
            blu_work_im[k];


        const float xr =
            cr *
            blu_chirp_re[k] +
            ci *
            blu_chirp_im[k];


        const float xi =
            ci *
            blu_chirp_re[k] -
            cr *
            blu_chirp_im[k];


        power_out[k] =
            xr * xr +
            xi * xi;
    }
}


/* ============================================================
 * Slaney mel conversion
 * ============================================================ */

static inline float hz_to_mel_slaney(
    float hz)
{
    if (hz < 1000.0f) {

        return
            3.0f *
            hz /
            200.0f;
    }


    const float logstep =
        logf(6.4f) /
        27.0f;


    return
        15.0f +
        logf(
            hz /
            1000.0f) /
        logstep;
}


static inline float mel_to_hz_slaney(
    float mel)
{
    if (mel < 15.0f) {

        return
            (200.0f / 3.0f) *
            mel;
    }


    const float logstep =
        logf(6.4f) /
        27.0f;


    return
        1000.0f *
        expf(
            (mel - 15.0f) *
            logstep);
}


/* ============================================================
 * Compute one mel frame
 * ============================================================ */

static void compute_mel_frame(
    const int16_t *samples,
    float *out_mel)
{
    static float hann_window[N_FFT];

    static float windowed[N_FFT];

    static float power_spec[
        N_FFT / 2 + 1];


    static float center_freq[
        N_MELS + 2];


    static int left_bin[N_MELS];

    static int center_bin[N_MELS];

    static int right_bin[N_MELS];


    static bool window_ready =
        false;

    static bool mel_ready =
        false;


    /* --------------------------------------------------------
     * Hann window
     * -------------------------------------------------------- */

    if (!window_ready) {

        for (int n = 0;
             n < N_FFT;
             ++n) {

            hann_window[n] =
                0.5f -
                0.5f *
                cosf(
                    2.0f *
                    (float)M_PI *
                    (float)n /
                    (float)N_FFT);
        }

        window_ready =
            true;
    }


    /* --------------------------------------------------------
     * Mel filter geometry
     * -------------------------------------------------------- */

    if (!mel_ready) {

        const float mel_lo =
            hz_to_mel_slaney(
                20.0f);

        const float mel_hi =
            hz_to_mel_slaney(
                7600.0f);


        for (int i = 0;
             i < N_MELS + 2;
             ++i) {

            const float mel =
                mel_lo +
                (mel_hi - mel_lo) *
                (float)i /
                (float)(N_MELS + 1);


            center_freq[i] =
                mel_to_hz_slaney(
                    mel);
        }


        const float bin_hz =
            (float)SAMPLE_RATE /
            (float)N_FFT;


        for (int m = 0;
             m < N_MELS;
             ++m) {

            left_bin[m] =
                (int)ceilf(
                    center_freq[m] /
                    bin_hz);

            center_bin[m] =
                (int)floorf(
                    center_freq[m + 1] /
                    bin_hz);

            right_bin[m] =
                (int)floorf(
                    center_freq[m + 2] /
                    bin_hz);


            if (left_bin[m] < 0)
                left_bin[m] = 0;

            if (center_bin[m] < 0)
                center_bin[m] = 0;

            if (right_bin[m] >
                N_FFT / 2) {

                right_bin[m] =
                    N_FFT / 2;
            }
        }


        mel_ready =
            true;
    }


    /* --------------------------------------------------------
     * Apply Hann window
     * -------------------------------------------------------- */

    for (int n = 0;
         n < N_FFT;
         ++n) {

        windowed[n] =
            (float)samples[n] *
            hann_window[n];
    }


    /* --------------------------------------------------------
     * Exact 400-point spectrum
     * -------------------------------------------------------- */

    compute_power_spectrum_400(
        windowed,
        power_spec);


    /* --------------------------------------------------------
     * Mel filters
     * -------------------------------------------------------- */

    const float bin_hz =
        (float)SAMPLE_RATE /
        (float)N_FFT;


    for (int m = 0;
         m < N_MELS;
         ++m) {

        const float left =
            center_freq[m];

        const float center =
            center_freq[m + 1];

        const float right =
            center_freq[m + 2];


        float sum =
            0.0f;


        /* Rising side */

        for (int k =
                left_bin[m];
             k <= center_bin[m] &&
             k <= N_FFT / 2;
             ++k) {

            const float hz =
                (float)k *
                bin_hz;


            float w =
                (hz - left) /
                (center - left);


            if (w < 0.0f)
                w = 0.0f;

            if (w > 1.0f)
                w = 1.0f;


            sum +=
                w *
                power_spec[k];
        }


        /* Falling side */

        for (int k =
                center_bin[m] + 1;
             k <= right_bin[m] &&
             k <= N_FFT / 2;
             ++k) {

            const float hz =
                (float)k *
                bin_hz;


            float w =
                (right - hz) /
                (right - center);


            if (w < 0.0f)
                w = 0.0f;

            if (w > 1.0f)
                w = 1.0f;


            sum +=
                w *
                power_spec[k];
        }


        const float enorm =
            2.0f /
            (right - left);


        out_mel[m] =
            sum *
            enorm;
    }
}


/* ============================================================
 * V6 feature extraction
 * ============================================================ */

static void extract_v6_features(
    const int16_t *audio)
{
    static int16_t frame[N_FFT];

    static float mel[N_MELS];


    const int pad =
        N_FFT / 2;


    /* --------------------------------------------------------
     * Generate 101 frames
     * -------------------------------------------------------- */

    for (int frame_idx = 0;
         frame_idx < N_FRAMES;
         ++frame_idx) {

        const int start =
            frame_idx *
            HOP_LENGTH -
            pad;


        for (int n = 0;
             n < N_FFT;
             ++n) {

            const int idx =
                start + n;


            /*
             * librosa center=True,
             * pad_mode='constant'
             */

            if (idx < 0 ||
                idx >= WINDOW_SAMPLES) {

                frame[n] =
                    0;

            } else {

                frame[n] =
                    audio[idx];
            }
        }


        compute_mel_frame(
            frame,
            mel);


        for (int m = 0;
             m < N_MELS;
             ++m) {

            float power =
                mel[m];


            if (power <
                1.0e-10f) {

                power =
                    1.0e-10f;
            }


            mel_db[
                m *
                N_FRAMES +
                frame_idx] =

                10.0f *
                log10f(power);
        }
    }


    /* --------------------------------------------------------
     * power_to_db(ref=np.max)
     * -------------------------------------------------------- */

    float max_db =
        mel_db[0];


    const int total =
        N_MELS *
        N_FRAMES;


    for (int i = 1;
         i < total;
         ++i) {

        if (mel_db[i] >
            max_db) {

            max_db =
                mel_db[i];
        }
    }


    /*
     * librosa top_db = 80
     */

    const float min_db =
        max_db - 80.0f;


    float sum =
        0.0f;


    for (int i = 0;
         i < total;
         ++i) {

        if (mel_db[i] <
            min_db) {

            mel_db[i] =
                min_db;
        }


        mel_db[i] -=
            max_db;


        sum +=
            mel_db[i];
    }


    /* --------------------------------------------------------
     * Standardization
     * -------------------------------------------------------- */

    const float mean =
        sum /
        (float)total;


    float variance =
        0.0f;


    for (int i = 0;
         i < total;
         ++i) {

        const float d =
            mel_db[i] -
            mean;


        variance +=
            d * d;
    }


    float std =
        sqrtf(
            variance /
            (float)total);


    if (std <
        1.0e-6f) {

        std =
            1.0e-6f;
    }


    /* --------------------------------------------------------
     * INT8 quantization
     * -------------------------------------------------------- */

    for (int i = 0;
         i < total;
         ++i) {

        feature[i] =
            (mel_db[i] -
             mean) /
            std;


        int q =
            (int)lroundf(
                feature[i] /
                INPUT_SCALE);


        q +=
            INPUT_ZERO_POINT;


        if (q < -128)
            q = -128;

        if (q > 127)
            q = 127;


        input_quant[i] =
            (int8_t)q;
    }
}


/* ============================================================
 * V6 inference
 * ============================================================ */

static float run_v6_inference(void)
{
    memcpy(
        input_tensor->data.int8,
        input_quant,
        N_MELS *
        N_FRAMES *
        sizeof(int8_t));


    if (interpreter->Invoke() !=
        kTfLiteOk) {

        ESP_LOGE(
            TAG,
            "V6 Invoke() failed");

        return 0.0f;
    }


    const int q =
        output_tensor->
            data.int8[0];


    return
        ((float)q -
         (float)OUTPUT_ZERO_POINT) *
        OUTPUT_SCALE;
}


/* ============================================================
 * Read I2S samples
 * ============================================================ */

static void read_samples(
    int16_t *dst,
    int count)
{
    int filled =
        0;


    int32_t raw[256];


    while (filled < count) {

        const int want =
            (count - filled > 256)
                ? 256
                : count - filled;


        size_t bytes_read =
            0;


        ESP_ERROR_CHECK(
            i2s_channel_read(
                rx_handle,
                raw,
                want *
                    sizeof(int32_t),
                &bytes_read,
                portMAX_DELAY));


        const int got =
            (int)(
                bytes_read /
                sizeof(int32_t));


        for (int i = 0;
             i < got &&
             filled < count;
             ++i) {

            dst[filled++] =
                (int16_t)(
                    raw[i] >>
                    14);
        }
    }
}


/* ============================================================
 * Initialize INMP441
 * ============================================================ */

static void init_microphone(void)
{
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            I2S_NUM_0,
            I2S_ROLE_MASTER);


    ESP_ERROR_CHECK(
        i2s_new_channel(
            &chan_cfg,
            NULL,
            &rx_handle));


    i2s_std_config_t std_cfg = {

        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(
                SAMPLE_RATE),

        .slot_cfg =
            I2S_STD_MSB_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO),

        .gpio_cfg = {

            .mclk =
                I2S_GPIO_UNUSED,

            .bclk =
                I2S_BCLK,

            .ws =
                I2S_WS,

            .dout =
                I2S_GPIO_UNUSED,

            .din =
                I2S_SD,

            .invert_flags = {

                .mclk_inv = 0,

                .bclk_inv = 0,

                .ws_inv = 0
            }
        }
    };


    std_cfg.slot_cfg.slot_mask =
        I2S_STD_SLOT_LEFT;


    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(
            rx_handle,
            &std_cfg));


    ESP_ERROR_CHECK(
        i2s_channel_enable(
            rx_handle));
}


/* ============================================================
 * Initialize V6 TFLite Micro
 * ============================================================ */

static void init_v6_model(void)
{
    const tflite::Model *model =
        tflite::GetModel(
            g_echoedge_v6_model);


    if (model->version() !=
        TFLITE_SCHEMA_VERSION) {

        ESP_LOGE(
            TAG,
            "TFLite schema mismatch: "
            "model=%d runtime=%d",
            model->version(),
            TFLITE_SCHEMA_VERSION);

        abort();
    }


    static constexpr size_t
        TENSOR_ARENA_SIZE =
            160 * 1024;


    static uint8_t *
        tensor_arena = NULL;


    tensor_arena =
        (uint8_t *)
        heap_caps_malloc(
            TENSOR_ARENA_SIZE,
            MALLOC_CAP_8BIT |
            MALLOC_CAP_SPIRAM);


    if (!tensor_arena) {

        ESP_LOGE(
            TAG,
            "Tensor arena allocation failed");

        abort();
    }


    static
    tflite::MicroMutableOpResolver<6>
        resolver;


    ESP_ERROR_CHECK(
        resolver.AddConv2D());

    ESP_ERROR_CHECK(
        resolver.AddDepthwiseConv2D());

    ESP_ERROR_CHECK(
        resolver.AddMaxPool2D());

    ESP_ERROR_CHECK(
        resolver.AddMean());

    ESP_ERROR_CHECK(
        resolver.AddFullyConnected());

    ESP_ERROR_CHECK(
        resolver.AddLogistic());


    static tflite::MicroInterpreter
        static_interpreter(
            model,
            resolver,
            tensor_arena,
            TENSOR_ARENA_SIZE,
            nullptr,
            nullptr,
            false);


    interpreter =
        &static_interpreter;


    if (interpreter->AllocateTensors() !=
        kTfLiteOk) {

        ESP_LOGE(
            TAG,
            "AllocateTensors() failed");

        abort();
    }


    input_tensor =
        interpreter->input(0);

    output_tensor =
        interpreter->output(0);


    ESP_LOGI(
        TAG,
        "V6 model ready: "
        "input type=%d "
        "shape=%d x %d x %d x %d "
        "scale=%.9f zp=%d",

        (int)input_tensor->type,

        input_tensor->dims->data[0],
        input_tensor->dims->data[1],
        input_tensor->dims->data[2],
        input_tensor->dims->data[3],

        input_tensor->params.scale,

        input_tensor->
            params.zero_point);


    ESP_LOGI(
        TAG,
        "V6 output type=%d "
        "scale=%.9f zp=%d",

        (int)output_tensor->type,

        output_tensor->params.scale,

        output_tensor->
            params.zero_point);
}
static void print_memory_usage(const char *stage)
{
    size_t internal_free =
        heap_caps_get_free_size(MALLOC_CAP_INTERNAL);

    size_t internal_largest =
        heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL);

    size_t psram_free =
        heap_caps_get_free_size(MALLOC_CAP_SPIRAM);

    size_t psram_largest =
        heap_caps_get_largest_free_block(MALLOC_CAP_SPIRAM);

    ESP_LOGI(
        TAG,
        "[RAM] %s | internal free=%u KB | internal largest=%u KB | PSRAM free=%u KB | PSRAM largest=%u KB",
        stage,
        (unsigned)(internal_free / 1024),
        (unsigned)(internal_largest / 1024),
        (unsigned)(psram_free / 1024),
        (unsigned)(psram_largest / 1024)
    );
}


/* ============================================================
 * Main application
 * ============================================================ */

extern "C" void app_main(void)
{
    ESP_LOGI(
        TAG,
        "V6 ECHOEDGE "
        "optimized exact-400FFT "
        "pipeline starting");
    print_memory_usage("startup");

    /* --------------------------------------------------------
     * LED GPIO7
     * -------------------------------------------------------- */

    gpio_config_t led_cfg =
        {};


    led_cfg.pin_bit_mask =
        1ULL <<
        LED_GPIO;

    led_cfg.mode =
        GPIO_MODE_OUTPUT;

    led_cfg.pull_down_en =
        GPIO_PULLDOWN_DISABLE;

    led_cfg.pull_up_en =
        GPIO_PULLUP_DISABLE;

    led_cfg.intr_type =
        GPIO_INTR_DISABLE;


    ESP_ERROR_CHECK(
        gpio_config(
            &led_cfg));


    gpio_set_level(
        LED_GPIO,
        0);


    /* --------------------------------------------------------
     * Allocate feature buffers
     * -------------------------------------------------------- */

    audio_window =
        (int16_t *)
        heap_caps_malloc(
            WINDOW_SAMPLES *
            sizeof(int16_t),
            MALLOC_CAP_8BIT |
            MALLOC_CAP_SPIRAM);


    mel_db =
        (float *)
        heap_caps_malloc(
            N_MELS *
            N_FRAMES *
            sizeof(float),
            MALLOC_CAP_8BIT |
            MALLOC_CAP_SPIRAM);


    feature =
        (float *)
        heap_caps_malloc(
            N_MELS *
            N_FRAMES *
            sizeof(float),
            MALLOC_CAP_8BIT |
            MALLOC_CAP_SPIRAM);


    input_quant =
        (int8_t *)
        heap_caps_malloc(
            N_MELS *
            N_FRAMES *
            sizeof(int8_t),
            MALLOC_CAP_8BIT |
            MALLOC_CAP_SPIRAM);


    if (!audio_window ||
        !mel_db ||
        !feature ||
        !input_quant) {

        ESP_LOGE(
            TAG,
            "Feature buffer allocation failed");

        abort();
    }


    /* --------------------------------------------------------
     * Microphone
     * -------------------------------------------------------- */

    init_microphone();


    /* --------------------------------------------------------
     * Initialize exact 400-point FFT
     * -------------------------------------------------------- */

    if (!init_400_fft_tables()) {

        abort();
    }


    /* --------------------------------------------------------
     * Initialize V6 model
     * -------------------------------------------------------- */

    init_v6_model();

    print_memory_usage("after V6 model");


    /* --------------------------------------------------------
     * Collect first 1 second
     * -------------------------------------------------------- */

    ESP_LOGI(
        TAG,
        "Collecting first "
        "1.0 second of microphone audio...");


    read_samples(
        audio_window,
        WINDOW_SAMPLES);


    uint32_t
        last_wake_ms = 0;
    print_memory_usage("before inference loop");

    /* --------------------------------------------------------
     * Continuous operation
     *
     * 1 second window
     * 0.5 second hop
     * -------------------------------------------------------- */

    while (true) {

        const int64_t t0 =
            esp_timer_get_time();


        extract_v6_features(
            audio_window);


        const int64_t t1 =
            esp_timer_get_time();


        const float probability =
            run_v6_inference();


        const int64_t t2 =
            esp_timer_get_time();


        const int64_t
            feature_ms =
                (t1 - t0) / 1000;


        const int64_t
            inference_ms =
                (t2 - t1) / 1000;


        ESP_LOGI(
            TAG,
            "V6 feature_ms=%lld "
            "inference_ms=%lld "
            "p=%.4f",

            (long long)
                feature_ms,

            (long long)
                inference_ms,

            probability);


        const uint32_t
            now_ms =
                (uint32_t)(
                    xTaskGetTickCount() *
                    portTICK_PERIOD_MS);


        /* ----------------------------------------------------
         * ECHOEDGE detection
         * ---------------------------------------------------- */

        if (probability >=
                WAKE_THRESHOLD &&

            (now_ms -
             last_wake_ms >=
             LED_HOLD_MS)) {

            last_wake_ms =
                now_ms;


            ESP_LOGW(
                TAG,
                ">>> ECHOEDGE DETECTED <<<");


            gpio_set_level(
                LED_GPIO,
                1);


            vTaskDelay(
                pdMS_TO_TICKS(
                    LED_HOLD_MS));


            gpio_set_level(
                LED_GPIO,
                0);
        }


        /* ----------------------------------------------------
         * Shift window by 0.5 second
         * ---------------------------------------------------- */

        memmove(
            audio_window,

            audio_window +
                HOP_SAMPLES,

            (WINDOW_SAMPLES -
             HOP_SAMPLES) *
            sizeof(int16_t));


        read_samples(
            audio_window +
                (WINDOW_SAMPLES -
                 HOP_SAMPLES),

            HOP_SAMPLES);
    }
}