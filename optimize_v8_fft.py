from pathlib import Path

p = Path("main/v6_echoedge.cc")
s = p.read_text()

# ------------------------------------------------------------
# Add precomputed FFT tables
# ------------------------------------------------------------
old = """static bool blu_fft_ready = false;"""

new = """static bool blu_fft_ready = false;

/*
 * Precomputed 1024-point FFT tables.
 * These remove repeated bit-reversal calculations and
 * per-butterfly twiddle-angle generation.
 */
static uint16_t blu_bitrev[BLU_FFT_SIZE];
static float blu_twiddle_re[BLU_FFT_SIZE / 2];
static float blu_twiddle_im[BLU_FFT_SIZE / 2];
"""

if old not in s:
    raise RuntimeError("Could not find blu_fft_ready")

s = s.replace(old, new, 1)

# ------------------------------------------------------------
# Replace radix-2 FFT
# ------------------------------------------------------------
start_marker = """/* ============================================================
 * Radix-2 complex FFT
 * ============================================================ */
"""

end_marker = """/* ============================================================
 * Initialize exact 400-point Bluestein FFT
 * ============================================================ */
"""

start = s.find(start_marker)
end = s.find(end_marker)

if start == -1 or end == -1 or end <= start:
    raise RuntimeError("Could not locate radix2_fft section")

new_fft = r'''/* ============================================================
 * Optimized radix-2 complex FFT
 *
 * Uses:
 *   - precomputed bit reversal
 *   - precomputed 1024-point twiddle factors
 *
 * The FFT size used by Bluestein is fixed at 1024.
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

    for (int i = 0; i < n; ++i) {

        const int j = blu_bitrev[i];

        if (i < j) {

            const float tr = re[i];
            re[i] = re[j];
            re[j] = tr;

            const float ti = im[i];
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

        const int half = len >> 1;
        const int twiddle_step = BLU_FFT_SIZE / len;

        for (int base = 0;
             base < n;
             base += len) {

            for (int j = 0;
                 j < half;
                 ++j) {

                const int u = base + j;
                const int v = u + half;

                const int tw = j * twiddle_step;

                float w_re = blu_twiddle_re[tw];
                float w_im = blu_twiddle_im[tw];

                if (inverse) {
                    w_im = -w_im;
                }

                const float v_re =
                    re[v] * w_re -
                    im[v] * w_im;

                const float v_im =
                    re[v] * w_im +
                    im[v] * w_re;

                const float u_re = re[u];
                const float u_im = im[u];

                re[u] = u_re + v_re;
                im[u] = u_im + v_im;

                re[v] = u_re - v_re;
                im[v] = u_im - v_im;
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


'''

s = s[:start] + new_fft + s[end:]

# ------------------------------------------------------------
# Add table initialization at beginning of init_400_fft_tables
# ------------------------------------------------------------
needle = """static bool init_400_fft_tables(void)
{
    const int N =
        BLU_N;

    const int M =
        BLU_FFT_SIZE;
"""

replacement = """static bool init_400_fft_tables(void)
{
    const int N =
        BLU_N;

    const int M =
        BLU_FFT_SIZE;

    /* --------------------------------------------------------
     * Precompute 1024-point bit reversal
     * -------------------------------------------------------- */

    for (int i = 0; i < M; ++i) {

        uint16_t x = (uint16_t)i;
        uint16_t r = 0;

        for (int b = 0; b < 10; ++b) {
            r = (uint16_t)((r << 1) | (x & 1U));
            x >>= 1;
        }

        blu_bitrev[i] = r;
    }

    /* --------------------------------------------------------
     * Precompute 1024-point FFT twiddles
     * -------------------------------------------------------- */

    for (int k = 0; k < M / 2; ++k) {

        const float angle =
            -2.0f *
            (float)M_PI *
            (float)k /
            (float)M;

        blu_twiddle_re[k] = cosf(angle);
        blu_twiddle_im[k] = sinf(angle);
    }
"""

if needle not in s:
    raise RuntimeError("Could not find init_400_fft_tables header")

s = s.replace(needle, replacement, 1)

p.write_text(s, newline="\n")

print("FFT optimization applied successfully.")
print("File:", p)
