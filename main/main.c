#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_log.h"
#include "esp_err.h"

#include "driver/i2s_std.h"

#include "esp_wn_iface.h"
#include "esp_wn_models.h"
#include "model_path.h"

#define I2S_PORT        0
#define I2S_SCK_GPIO    4
#define I2S_WS_GPIO     5
#define I2S_SD_GPIO     6

#define SAMPLE_RATE     16000

static const char *TAG = "ECHOEDGE_TEST";

static i2s_chan_handle_t rx_handle = NULL;

static void microphone_init(void)
{
    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(I2S_PORT, I2S_ROLE_MASTER);

    ESP_ERROR_CHECK(
        i2s_new_channel(&chan_cfg, NULL, &rx_handle)
    );

    i2s_std_config_t std_cfg = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                I2S_DATA_BIT_WIDTH_32BIT,
                I2S_SLOT_MODE_MONO
            ),

        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_SCK_GPIO,
            .ws   = I2S_WS_GPIO,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_SD_GPIO,

            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false
            }
        }
    };

    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(rx_handle, &std_cfg)
    );

    ESP_ERROR_CHECK(
        i2s_channel_enable(rx_handle)
    );

    ESP_LOGI(TAG, "INMP441 initialized");
}

static size_t microphone_read(int16_t *output, size_t samples)
{
    int32_t *raw =
        malloc(samples * sizeof(int32_t));

    if (raw == NULL) {
        ESP_LOGE(TAG, "Microphone buffer allocation failed");
        return 0;
    }

    size_t bytes_read = 0;

    esp_err_t err = i2s_channel_read(
        rx_handle,
        raw,
        samples * sizeof(int32_t),
        &bytes_read,
        portMAX_DELAY
    );

    if (err != ESP_OK) {
        free(raw);
        return 0;
    }

    size_t count =
        bytes_read / sizeof(int32_t);

    for (size_t i = 0; i < count; i++) {

        /*
         * INMP441 provides 24-bit audio
         * inside a 32-bit I2S frame.
         *
         * Convert to signed 16-bit PCM.
         */
        output[i] =
            (int16_t)(raw[i] >> 14);
    }

    free(raw);

    return count;
}

void app_main(void)
{
    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, " EchoEdge WakeNet Test");
    ESP_LOGI(TAG, " ESP32-S3 + INMP441");
    ESP_LOGI(TAG, "================================");

    /*
     * IMPORTANT:
     * Do NOT initialize the microphone yet.
     * We first test whether WakeNet itself
     * can be created successfully.
     */

    ESP_LOGI(TAG, "Loading WakeNet models...");

    srmodel_list_t *models =
        esp_srmodel_init("model");

    if (models == NULL) {
        ESP_LOGE(TAG, "Could not load model partition!");
        ESP_LOGE(TAG, "WakeNet cannot start.");
        return;
    }

    char *model_name =
        esp_srmodel_filter(
            models,
            ESP_WN_PREFIX,
            NULL
        );

    if (model_name == NULL) {
        ESP_LOGE(TAG, "No WakeNet model found!");
        esp_srmodel_deinit(models);
        return;
    }

    ESP_LOGI(TAG, "WakeNet model: %s", model_name);

    char *wake_word =
        esp_srmodel_get_wake_words(
            models,
            model_name
        );

    if (wake_word != NULL) {
        ESP_LOGI(TAG, "Wake word: %s", wake_word);
    }

    const esp_wn_iface_t *wakenet =
        esp_wn_handle_from_name(model_name);

    if (wakenet == NULL) {
        ESP_LOGE(TAG, "Could not create WakeNet interface!");
        esp_srmodel_deinit(models);
        return;
    }

    ESP_LOGI(TAG, "Creating WakeNet model...");

    model_iface_data_t *model_data =
        wakenet->create(
            model_name,
            DET_MODE_95
        );

    if (model_data == NULL) {
        ESP_LOGE(TAG, "WakeNet model creation failed!");
        esp_srmodel_deinit(models);
        return;
    }

    ESP_LOGI(TAG, "WakeNet model created successfully!");

    int wake_rate =
        wakenet->get_samp_rate(model_data);

    int wake_samples =
        wakenet->get_samp_chunksize(model_data);

    ESP_LOGI(
        TAG,
        "WakeNet sample rate: %d Hz",
        wake_rate
    );

    ESP_LOGI(
        TAG,
        "WakeNet chunk size: %d samples",
        wake_samples
    );

    /*
     * WakeNet has now successfully initialized.
     * Only after this point do we initialize
     * the INMP441 microphone.
     */
    microphone_init();

    int16_t *audio_buffer =
        malloc(wake_samples * sizeof(int16_t));

    if (audio_buffer == NULL) {
        ESP_LOGE(TAG, "WakeNet audio buffer allocation failed!");
        wakenet->destroy(model_data);
        esp_srmodel_deinit(models);
        return;
    }

    ESP_LOGI(TAG, "================================");
    ESP_LOGI(TAG, " WAKE WORD LISTENER ACTIVE");
    ESP_LOGI(TAG, " Speak the built-in wake word");
    ESP_LOGI(TAG, "================================");

    while (1) {

        size_t received =
            microphone_read(
                audio_buffer,
                wake_samples
            );

        if (received != wake_samples) {
            continue;
        }

        wakenet_state_t state =
            wakenet->detect(
                model_data,
                audio_buffer
            );

        if (state == WAKENET_DETECTED) {

            ESP_LOGI(
                TAG,
                "********************************"
            );

            ESP_LOGI(
                TAG,
                "*** WAKE WORD DETECTED! ***"
            );

            ESP_LOGI(
                TAG,
                "********************************"
            );
        }
    }
}