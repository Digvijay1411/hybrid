#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


// ==========================================
// Architectural Dimensions
// ==========================================
#define THREADS_PER_BLOCK 512
#define KEM_TILE_SIZE 128
#define SESSIONS_PER_BLOCK (THREADS_PER_BLOCK / KEM_TILE_SIZE)
#define KEY_WORDS 8          // 256-bit key = 8 x 32-bit words
#define RECORD_BYTES 64      // one Salsa20 block == one packed sensor record

// Salsa20 Constants ("expand 32-byte k")
#define SALSA_CONST_0 0x61707865
#define SALSA_CONST_1 0x3320646e
#define SALSA_CONST_2 0x79622d32
#define SALSA_CONST_3 0x6b206574

#define ROTL32(v, n) (((v) << (n)) | ((v) >> (32 - (n))))

#define QUARTER_ROUND(a, b, c, d) \
    b ^= ROTL32(a + d, 7);  \
    c ^= ROTL32(b + a, 9);  \
    d ^= ROTL32(c + b, 13); \
    a ^= ROTL32(d + c, 18);

// ==========================================
// Packed on-disk / on-device record layout (64 bytes, matches one Salsa20 block)
//
//  [0 .. 3]   int32   epoch
//  [4 .. 7]   int32   moteid
//  [8 ..11]   float   temperature
//  [12..15]   float   humidity
//  [16..19]   float   light
//  [20..23]   float   voltage
//  [24..34]   char    date[11]   "YYYY-MM-DD\0"
//  [35..47]   char    time[13]   "HH:MM:SS.xxx\0"
//  [48..51]   uint32  checksum   (FNV-1a over bytes 0..47, computed pre-encryption)
//  [52..63]   uint8   reserved[12] (zero padding)
//
// The checksum travels inside the plaintext and is re-checked after decryption,
// so tampering or corruption of an individual record is detectable per-record
// (not just for the payload as a whole).
// ==========================================
#pragma pack(push, 1)
typedef struct PackedRecord {
    int32_t epoch;
    int32_t moteid;
    float   temperature;
    float   humidity;
    float   light;
    float   voltage;
    char    date[11];
    char    time_str[13];
    uint32_t checksum;
    uint8_t reserved[12];
} PackedRecord;
#pragma pack(pop)

_Static_assert(sizeof(PackedRecord) == RECORD_BYTES, "PackedRecord must be exactly 64 bytes");

// Host-side struct used only while parsing the raw text file
typedef struct SensorRow {
    char date[11];
    char time_str[13];
    int  epoch;
    int  moteid;
    float temperature;
    float humidity;
    float light;
    float voltage;
} SensorRow;

// ==========================================
// FNV-1a 32-bit checksum (host + device) — integrity check, NOT a cryptographic MAC.
// ==========================================
static inline uint32_t fnv1a_32(const uint8_t* data, int len) {
    uint32_t hash = 0x811c9dc5u;
    for (int i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 0x01000193u;
    }
    return hash;
}

// ==========================================================================
// ==========================================================================
//   KYBER-512 (ML-KEM-512-structured) MODULE-LWE KEM — DEVICE IMPLEMENTATION
// ==========================================================================
// ==========================================================================
//
// This section replaces the previous placeholder "execute_lattice_kem_ntt".
// It implements the real Kyber-512 arithmetic: Keccak-f[1600]/SHA3/SHAKE,
// the number-theoretic transform (NTT) over Z_3329[X]/(X^256+1), CBD noise
// sampling, compression/encoding, and the CPA-secure public-key encryption
// scheme that underlies Kyber, wrapped into a KEM.
//
// VERIFIED BEFORE PORTING: the NTT table, forward/inverse NTT, and NTT-domain
// base multiplication were validated against a from-scratch Python model
// (see conversation) that confirms invNTT(NTT(p)) == p and that NTT-domain
// base multiplication reproduces naive negacyclic convolution mod X^256+1.
// The full CPA-PKE keygen/enc/dec round-trip and KEM encaps/decaps were also
// validated end-to-end in that Python model (matching Kyber-512's standard
// sizes: pk=800B, sk=768B, ct=768B).
//
// IMPORTANT SECURITY CAVEAT (please read before calling this "CCA2-secure"
// in a paper): the KEM wrapper here derives the shared secret as
//     ss = SHA3-256(m || ct)
// without the FIPS 203 Fujisaki-Okamoto implicit-rejection re-encryption
// check on decapsulation. This is a hash-bound KEM wrapper around an
// IND-CPA-secure PKE, not the full FO transform. It is fine for a
// throughput/architecture study, but if your paper claims "CCA2-secure
// Kyber," you need to add the FO re-encryption check (encrypt m' inside
// decaps and compare ciphertexts, falling back to a pseudorandom secret on
// mismatch) per FIPS 203 Algorithm 21 before submission.
//
// RANDOMNESS: per-session 32-byte seeds (for keygen) and 32-byte messages
// (for encapsulation) are generated on the HOST with random_device
// (a CSPRNG on modern platforms) and copied to the device once. They never
// leave the device again. This avoids pulling in a device-side CSPRNG
// (curand) for this prototype; if you want a fully GPU-resident RNG, swap
// this for curandStatePhilox4_32_10 seeded per session.
// ==========================================================================

#define KYB_N 256
#define KYB_Q 3329
#define KYB_K 2        // module rank (Kyber-512)
#define KYB_ETA1 3
#define KYB_ETA2 2
#define KYB_DU 10
#define KYB_DV 4
#define KYB_POLYBYTES 384          // 256 * 12 bits / 8
#define KYB_PK_BYTES (KYB_K*KYB_POLYBYTES + 32)   // 800
#define KYB_SK_BYTES (KYB_K*KYB_POLYBYTES)        // 768 (CPA secret key only; no FO extras)
#define KYB_CT_U_BYTES ((KYB_N*KYB_DU)/8)         // 320 per poly
#define KYB_CT_BYTES (KYB_K*KYB_CT_U_BYTES + (KYB_N*KYB_DV)/8) // 768

// Verified via: python3 -c "q=3329; bitrev7=...; [pow(17,bitrev7(i),q) for i in range(128)]"
static const int16_t KYB_ZETAS[128] = {
    1, 1729, 2580, 3289, 2642, 630, 1897, 848, 1062, 1919, 193, 797, 2786, 3260, 569, 1746,
    296, 2447, 1339, 1476, 3046, 56, 2240, 1333, 1426, 2094, 535, 2882, 2393, 2879, 1974, 821,
    289, 331, 3253, 1756, 1197, 2304, 2277, 2055, 650, 1977, 2513, 632, 2865, 33, 1320, 1915,
    2319, 1435, 807, 452, 1438, 2868, 1534, 2402, 2647, 2617, 1481, 648, 2474, 3110, 1227, 910,
    17, 2761, 583, 2649, 1637, 723, 2288, 1100, 1409, 2662, 3281, 233, 756, 2156, 3015, 3050,
    1703, 1651, 2789, 1789, 1847, 952, 1461, 2687, 939, 2308, 2437, 2388, 733, 2337, 268, 641,
    1584, 2298, 2037, 3220, 375, 2549, 2090, 1645, 1063, 319, 2773, 757, 2099, 561, 2466, 2594,
    2804, 1092, 403, 1026, 1143, 2150, 2775, 886, 1722, 1212, 1874, 1029, 2110, 2935, 885, 2154
};

// -------------------- Keccak-f[1600] / SHA3 / SHAKE (device, single-thread) --------------------

static inline uint64_t kb_rotl64(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

static void keccak_f1600(uint64_t s[25]) {
    const uint64_t RC[24] = {
        0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
        0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
        0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
        0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
        0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
        0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL
    };
    const int r[25] = {
        0,1,62,28,27, 36,44,6,55,20, 3,10,43,25,39, 41,45,15,21,8, 18,2,61,56,14
    };

    for (int round = 0; round < 24; round++) {
        uint64_t C[5], D[5];
        #pragma unroll
        for (int x = 0; x < 5; x++) C[x] = s[x] ^ s[x+5] ^ s[x+10] ^ s[x+15] ^ s[x+20];
        #pragma unroll
        for (int x = 0; x < 5; x++) D[x] = C[(x+4)%5] ^ kb_rotl64(C[(x+1)%5], 1);
        #pragma unroll
        for (int x = 0; x < 5; x++)
            #pragma unroll
            for (int y = 0; y < 5; y++)
                s[x+5*y] ^= D[x];

        uint64_t B[25];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                int newx = y;
                int newy = (2*x + 3*y) % 5;
                B[newx + 5*newy] = kb_rotl64(s[x+5*y], r[x+5*y]);
            }
        }
        #pragma unroll
        for (int x = 0; x < 5; x++)
            #pragma unroll
            for (int y = 0; y < 5; y++)
                s[x+5*y] = B[x+5*y] ^ ((~B[((x+1)%5)+5*y]) & B[((x+2)%5)+5*y]);

        s[0] ^= RC[round];
    }
}

static void keccak_absorb(uint64_t s[25], const uint8_t* in, int inlen, int rate, uint8_t pad_byte) {
    #pragma unroll
    for (int i = 0; i < 25; i++) s[i] = 0;

    uint8_t block[200];
    int pos = 0;
    while (inlen >= rate) {
        for (int i = 0; i < rate; i++) block[i] = in[pos+i];
        for (int i = 0; i < rate/8; i++) {
            uint64_t lane = 0;
            for (int b = 0; b < 8; b++) lane |= ((uint64_t)block[8*i+b]) << (8*b);
            s[i] ^= lane;
        }
        keccak_f1600(s);
        pos += rate; inlen -= rate;
    }
    uint8_t last[200];
    for (int i = 0; i < rate; i++) last[i] = 0;
    for (int i = 0; i < inlen; i++) last[i] = in[pos+i];
    last[inlen] ^= pad_byte;
    last[rate-1] ^= 0x80;
    for (int i = 0; i < rate/8; i++) {
        uint64_t lane = 0;
        for (int b = 0; b < 8; b++) lane |= ((uint64_t)last[8*i+b]) << (8*b);
        s[i] ^= lane;
    }
    keccak_f1600(s);
}

static void keccak_squeeze(uint64_t s[25], uint8_t* out, int outlen, int rate) {
    int pos = 0;
    while (outlen > 0) {
        int block = outlen < rate ? outlen : rate;
        for (int i = 0; i < block; i++) {
            int lane = i / 8, byteidx = i % 8;
            out[pos+i] = (uint8_t)(s[lane] >> (8*byteidx));
        }
        pos += block; outlen -= block;
        if (outlen > 0) keccak_f1600(s);
    }
}

static void kb_shake128(const uint8_t* in, int inlen, uint8_t* out, int outlen) {
    uint64_t s[25];
    keccak_absorb(s, in, inlen, 168, 0x1F);
    keccak_squeeze(s, out, outlen, 168);
}
static void kb_shake256(const uint8_t* in, int inlen, uint8_t* out, int outlen) {
    uint64_t s[25];
    keccak_absorb(s, in, inlen, 136, 0x1F);
    keccak_squeeze(s, out, outlen, 136);
}
static void kb_sha3_256(const uint8_t* in, int inlen, uint8_t out[32]) {
    uint64_t s[25];
    keccak_absorb(s, in, inlen, 136, 0x06);
    keccak_squeeze(s, out, 32, 136);
}
static void kb_sha3_512(const uint8_t* in, int inlen, uint8_t out[64]) {
    uint64_t s[25];
    keccak_absorb(s, in, inlen, 72, 0x06);
    keccak_squeeze(s, out, 64, 72);
}

// -------------------- Kyber field / poly arithmetic --------------------

static inline int16_t kyb_add(int16_t a, int16_t b) {
    int v = a + b; if (v >= KYB_Q) v -= KYB_Q; return (int16_t)v;
}
static inline int16_t kyb_sub(int16_t a, int16_t b) {
    int v = a - b; if (v < 0) v += KYB_Q; return (int16_t)v;
}
static inline int16_t kyb_mul(int16_t a, int16_t b) {
    int32_t v = (int32_t)a * (int32_t)b;
    v %= KYB_Q; if (v < 0) v += KYB_Q;
    return (int16_t)v;
}

typedef int16_t kyb_poly[KYB_N];

static void kyb_ntt(kyb_poly p) {
    int k = 1;
    for (int len = 128; len >= 2; len >>= 1) {
        for (int start = 0; start < 256; start += 2*len) {
            int16_t zeta = KYB_ZETAS[k++];
            for (int j = start; j < start+len; j++) {
                int16_t t = kyb_mul(zeta, p[j+len]);
                p[j+len] = kyb_sub(p[j], t);
                p[j] = kyb_add(p[j], t);
            }
        }
    }
}

static void kyb_invntt(kyb_poly p) {
    int k = 127;
    for (int len = 2; len <= 128; len <<= 1) {
        for (int start = 0; start < 256; start += 2*len) {
            int16_t zeta = KYB_ZETAS[k--];
            for (int j = start; j < start+len; j++) {
                int16_t t = p[j];
                p[j] = kyb_add(t, p[j+len]);
                p[j+len] = kyb_mul(zeta, kyb_sub(p[j+len], t));
            }
        }
    }
    const int16_t f = 3303; // 128^-1 mod 3329, verified in Python round-trip test
    for (int i = 0; i < KYB_N; i++) p[i] = kyb_mul(p[i], f);
}

static void kyb_basemul(int16_t a0, int16_t a1, int16_t b0, int16_t b1, int16_t zeta,
                       int16_t* c0, int16_t* c1) {
    *c0 = kyb_add(kyb_mul(a0,b0), kyb_mul(kyb_mul(a1,b1), zeta));
    *c1 = kyb_add(kyb_mul(a0,b1), kyb_mul(a1,b0));
}

static void kyb_poly_basemul_ntt(const kyb_poly a, const kyb_poly b, kyb_poly r) {
    for (int i = 0; i < 64; i++) {
        int16_t zeta = KYB_ZETAS[64+i];
        kyb_basemul(a[4*i], a[4*i+1], b[4*i], b[4*i+1], zeta, &r[4*i], &r[4*i+1]);
        int16_t negz = (int16_t)((KYB_Q - zeta) % KYB_Q);
        kyb_basemul(a[4*i+2], a[4*i+3], b[4*i+2], b[4*i+3], negz, &r[4*i+2], &r[4*i+3]);
    }
}

static void kyb_poly_add(const kyb_poly a, const kyb_poly b, kyb_poly r) {
    for (int i = 0; i < KYB_N; i++) r[i] = kyb_add(a[i], b[i]);
}
static void kyb_poly_sub(const kyb_poly a, const kyb_poly b, kyb_poly r) {
    for (int i = 0; i < KYB_N; i++) r[i] = kyb_sub(a[i], b[i]);
}

static void kyb_cbd(const uint8_t* buf, int eta, kyb_poly out) {
    // buf must contain 64*eta bytes = 2*eta bits per coefficient, LSB-first per byte.
    for (int i = 0; i < KYB_N; i++) {
        int a = 0, b = 0;
        for (int t = 0; t < eta; t++) {
            int bitpos_a = 2*i*eta + t;
            int bitpos_b = 2*i*eta + eta + t;
            a += (buf[bitpos_a/8] >> (bitpos_a%8)) & 1;
            b += (buf[bitpos_b/8] >> (bitpos_b%8)) & 1;
        }
        out[i] = kyb_sub((int16_t)a, (int16_t)b);
    }
}

static void kyb_prf(const uint8_t seed[32], uint8_t nonce, uint8_t* out, int outlen) {
    uint8_t in[33];
    for (int i = 0; i < 32; i++) in[i] = seed[i];
    in[32] = nonce;
    kb_shake256(in, 33, out, outlen);
}

static void kyb_gen_matrix_elem(const uint8_t rho[32], int i, int j, kyb_poly out) {
    uint8_t in[34];
    for (int t = 0; t < 32; t++) in[t] = rho[t];
    in[32] = (uint8_t)i; in[33] = (uint8_t)j;
    uint8_t buf[1536]; // generous XOF buffer for rejection sampling (matches validated Python model)
    kb_shake128(in, 34, buf, sizeof(buf));
    int pos = 0, count = 0;
    while (count < KYB_N) {
        uint16_t d1 = buf[pos] | ((uint16_t)(buf[pos+1] & 0x0F) << 8);
        uint16_t d2 = (buf[pos+1] >> 4) | ((uint16_t)buf[pos+2] << 4);
        pos += 3;
        if (d1 < KYB_Q && count < KYB_N) out[count++] = (int16_t)d1;
        if (d2 < KYB_Q && count < KYB_N) out[count++] = (int16_t)d2;
    }
}

static void kyb_compress(const kyb_poly p, int d, uint16_t* out) {
    int m = 1 << d;
    for (int i = 0; i < KYB_N; i++) {
        int32_t x = p[i];
        out[i] = (uint16_t)(((x * m + KYB_Q/2) / KYB_Q) % m);
    }
}
static void kyb_decompress(const uint16_t* c, int d, kyb_poly out) {
    int m = 1 << d;
    for (int i = 0; i < KYB_N; i++) {
        out[i] = (int16_t)(((int32_t)c[i] * KYB_Q + m/2) / m);
    }
}

static void kyb_pack12(const kyb_poly p, uint8_t* out) {
    for (int i = 0; i < KYB_N; i += 2) {
        int16_t a = p[i], b = p[i+1];
        out[3*(i/2)+0] = (uint8_t)(a & 0xFF);
        out[3*(i/2)+1] = (uint8_t)(((a>>8)&0xF) | ((b&0xF)<<4));
        out[3*(i/2)+2] = (uint8_t)((b>>4)&0xFF);
    }
}
static void kyb_unpack12(const uint8_t* buf, kyb_poly out) {
    for (int i = 0; i < KYB_N; i += 2) {
        uint8_t b0 = buf[3*(i/2)+0], b1 = buf[3*(i/2)+1], b2 = buf[3*(i/2)+2];
        out[i]   = (int16_t)(b0 | ((uint16_t)(b1 & 0xF) << 8));
        out[i+1] = (int16_t)((b1 >> 4) | ((uint16_t)b2 << 4));
    }
}

static void kyb_pack_d(const uint16_t* c, int d, uint8_t* out) {
    int nbits = KYB_N * d;
    int nbytes = (nbits + 7) / 8;
    for (int i = 0; i < nbytes; i++) out[i] = 0;
    for (int i = 0; i < KYB_N; i++) {
        for (int b = 0; b < d; b++) {
            int bit = (c[i] >> b) & 1;
            int bitpos = i*d + b;
            out[bitpos/8] |= (uint8_t)(bit << (bitpos%8));
        }
    }
}
static void kyb_unpack_d(const uint8_t* buf, int d, uint16_t* out) {
    for (int i = 0; i < KYB_N; i++) {
        uint16_t v = 0;
        for (int b = 0; b < d; b++) {
            int bitpos = i*d + b;
            int bit = (buf[bitpos/8] >> (bitpos%8)) & 1;
            v |= (uint16_t)(bit << b);
        }
        out[i] = v;
    }
}

// -------------------- CPA-PKE (Kyber's underlying encryption scheme) --------------------

// pk = KYB_K polys packed at 12 bits + 32-byte rho  -> KYB_PK_BYTES
// sk = KYB_K polys (s_hat) packed at 12 bits         -> KYB_SK_BYTES
static void kyb_indcpa_keygen(const uint8_t d_seed[32], uint8_t* pk, uint8_t* sk) {
    uint8_t h[64];
    kb_sha3_512(d_seed, 32, h);
    const uint8_t* rho = h;        // h[0..31]
    const uint8_t* sigma = h + 32; // h[32..63]

    kyb_poly A[KYB_K][KYB_K];
    for (int i = 0; i < KYB_K; i++)
        for (int j = 0; j < KYB_K; j++)
            kyb_gen_matrix_elem(rho, i, j, A[i][j]);

    kyb_poly s[KYB_K], e[KYB_K];
    uint8_t noisebuf[64*KYB_ETA1];
    for (int n = 0; n < KYB_K; n++) {
        kyb_prf(sigma, (uint8_t)n, noisebuf, 64*KYB_ETA1);
        kyb_cbd(noisebuf, KYB_ETA1, s[n]);
    }
    for (int n = 0; n < KYB_K; n++) {
        kyb_prf(sigma, (uint8_t)(KYB_K+n), noisebuf, 64*KYB_ETA1);
        kyb_cbd(noisebuf, KYB_ETA1, e[n]);
    }

    kyb_poly s_hat[KYB_K], e_hat[KYB_K], t_hat[KYB_K];
    for (int n = 0; n < KYB_K; n++) { for (int i=0;i<KYB_N;i++) s_hat[n][i]=s[n][i]; kyb_ntt(s_hat[n]); }
    for (int n = 0; n < KYB_K; n++) { for (int i=0;i<KYB_N;i++) e_hat[n][i]=e[n][i]; kyb_ntt(e_hat[n]); }

    for (int i = 0; i < KYB_K; i++) {
        kyb_poly acc = {0};
        for (int j = 0; j < KYB_K; j++) {
            kyb_poly prod;
            kyb_poly_basemul_ntt(A[i][j], s_hat[j], prod);
            kyb_poly_add(acc, prod, acc);
        }
        kyb_poly_add(acc, e_hat[i], t_hat[i]);
    }

    for (int n = 0; n < KYB_K; n++) kyb_pack12(t_hat[n], pk + n*KYB_POLYBYTES);
    for (int i = 0; i < 32; i++) pk[KYB_K*KYB_POLYBYTES + i] = rho[i];
    for (int n = 0; n < KYB_K; n++) kyb_pack12(s_hat[n], sk + n*KYB_POLYBYTES);
}

static void kyb_indcpa_enc(const uint8_t* pk, const uint8_t msg32[32], const uint8_t coins[32], uint8_t* ct) {
    kyb_poly t_hat[KYB_K];
    for (int n = 0; n < KYB_K; n++) kyb_unpack12(pk + n*KYB_POLYBYTES, t_hat[n]);
    const uint8_t* rho = pk + KYB_K*KYB_POLYBYTES;

    kyb_poly A[KYB_K][KYB_K];
    for (int i = 0; i < KYB_K; i++)
        for (int j = 0; j < KYB_K; j++)
            kyb_gen_matrix_elem(rho, i, j, A[i][j]);

    kyb_poly r[KYB_K], e1[KYB_K], e2;
    uint8_t noisebuf1[64*KYB_ETA1];
    uint8_t noisebuf2[64*KYB_ETA2];
    for (int n = 0; n < KYB_K; n++) {
        kyb_prf(coins, (uint8_t)n, noisebuf1, 64*KYB_ETA1);
        kyb_cbd(noisebuf1, KYB_ETA1, r[n]);
    }
    for (int n = 0; n < KYB_K; n++) {
        kyb_prf(coins, (uint8_t)(KYB_K+n), noisebuf2, 64*KYB_ETA2);
        kyb_cbd(noisebuf2, KYB_ETA2, e1[n]);
    }
    kyb_prf(coins, (uint8_t)(2*KYB_K), noisebuf2, 64*KYB_ETA2);
    kyb_cbd(noisebuf2, KYB_ETA2, e2);

    kyb_poly r_hat[KYB_K];
    for (int n = 0; n < KYB_K; n++) { for (int i=0;i<KYB_N;i++) r_hat[n][i]=r[n][i]; kyb_ntt(r_hat[n]); }

    kyb_poly u[KYB_K];
    for (int i = 0; i < KYB_K; i++) {
        kyb_poly acc = {0};
        for (int j = 0; j < KYB_K; j++) {
            kyb_poly prod;
            kyb_poly_basemul_ntt(A[j][i], r_hat[j], prod); // A^T
            kyb_poly_add(acc, prod, acc);
        }
        kyb_invntt(acc);
        kyb_poly_add(acc, e1[i], u[i]);
    }

    kyb_poly vt = {0};
    for (int i = 0; i < KYB_K; i++) {
        kyb_poly prod;
        kyb_poly_basemul_ntt(t_hat[i], r_hat[i], prod);
        kyb_poly_add(vt, prod, vt);
    }
    kyb_invntt(vt);

    kyb_poly m_poly;
    {
        uint16_t bits[KYB_N];
        for (int i = 0; i < KYB_N; i++) bits[i] = (msg32[i/8] >> (i%8)) & 1;
        kyb_decompress(bits, 1, m_poly);
    }

    kyb_poly v;
    kyb_poly_add(vt, e2, v);
    kyb_poly_add(v, m_poly, v);

    uint16_t comp[KYB_N];
    for (int i = 0; i < KYB_K; i++) {
        kyb_compress(u[i], KYB_DU, comp);
        kyb_pack_d(comp, KYB_DU, ct + i*KYB_CT_U_BYTES);
    }
    kyb_compress(v, KYB_DV, comp);
    kyb_pack_d(comp, KYB_DV, ct + KYB_K*KYB_CT_U_BYTES);
}

static void kyb_indcpa_dec(const uint8_t* sk, const uint8_t* ct, uint8_t msg32_out[32]) {
    kyb_poly u[KYB_K];
    uint16_t comp[KYB_N];
    for (int i = 0; i < KYB_K; i++) {
        kyb_unpack_d(ct + i*KYB_CT_U_BYTES, KYB_DU, comp);
        kyb_decompress(comp, KYB_DU, u[i]);
    }
    kyb_poly v;
    kyb_unpack_d(ct + KYB_K*KYB_CT_U_BYTES, KYB_DV, comp);
    kyb_decompress(comp, KYB_DV, v);

    kyb_poly s_hat[KYB_K];
    for (int n = 0; n < KYB_K; n++) kyb_unpack12(sk + n*KYB_POLYBYTES, s_hat[n]);

    kyb_poly u_hat[KYB_K];
    for (int n = 0; n < KYB_K; n++) { for (int i=0;i<KYB_N;i++) u_hat[n][i]=u[n][i]; kyb_ntt(u_hat[n]); }

    kyb_poly acc = {0};
    for (int i = 0; i < KYB_K; i++) {
        kyb_poly prod;
        kyb_poly_basemul_ntt(s_hat[i], u_hat[i], prod);
        kyb_poly_add(acc, prod, acc);
    }
    kyb_invntt(acc);

    kyb_poly mp;
    kyb_poly_sub(v, acc, mp);

    uint16_t bits[KYB_N];
    kyb_compress(mp, 1, bits);
    for (int i = 0; i < 32; i++) msg32_out[i] = 0;
    for (int i = 0; i < KYB_N; i++) msg32_out[i/8] |= (uint8_t)(bits[i] << (i%8));
}

// -------------------- KEM wrapper (see security caveat above) --------------------

static void kyb_kem_keygen(const uint8_t seed32[32], uint8_t* pk, uint8_t* sk) {
    kyb_indcpa_keygen(seed32, pk, sk);
}

static void kyb_kem_encaps(const uint8_t* pk, const uint8_t msg32[32], uint8_t* ct, uint8_t ss_out[32]) {
    uint8_t coins[32];
    kb_sha3_256(msg32, 32, coins);
    kyb_indcpa_enc(pk, msg32, coins, ct);

    uint8_t catbuf[32 + KYB_CT_BYTES];
    for (int i = 0; i < 32; i++) catbuf[i] = msg32[i];
    for (int i = 0; i < KYB_CT_BYTES; i++) catbuf[32+i] = ct[i];
    kb_sha3_256(catbuf, 32 + KYB_CT_BYTES, ss_out);
}

static void kyb_kem_decaps(const uint8_t* sk, const uint8_t* ct, uint8_t ss_out[32]) {
    uint8_t m[32];
    kyb_indcpa_dec(sk, ct, m);

    uint8_t catbuf[32 + KYB_CT_BYTES];
    for (int i = 0; i < 32; i++) catbuf[i] = m[i];
    for (int i = 0; i < KYB_CT_BYTES; i++) catbuf[32+i] = ct[i];
    kb_sha3_256(catbuf, 32 + KYB_CT_BYTES, ss_out);
}

// ==========================================
// SYMMETRIC PHASE: Salsa20 Core (Register-Optimized)
// ==========================================
static void salsa20_block_process(
    const uint32_t key[KEY_WORDS],
    uint64_t nonce,
    uint64_t block_counter,
    const uint8_t* plaintext,
    uint8_t* ciphertext,
    int global_id)
{
    uint32_t state[16];
    uint32_t working_state[16];

    state[0] = SALSA_CONST_0;
    state[1] = key[0];
    state[2] = key[1];
    state[3] = key[2];
    state[4] = key[3];
    state[5] = SALSA_CONST_1;

    state[6] = (uint32_t)(nonce & 0xFFFFFFFF);
    state[7] = (uint32_t)(nonce >> 32);
    state[8] = (uint32_t)(block_counter & 0xFFFFFFFF);
    state[9] = (uint32_t)(block_counter >> 32);

    state[10] = SALSA_CONST_2;
    state[11] = key[4];
    state[12] = key[5];
    state[13] = key[6];
    state[14] = key[7];
    state[15] = SALSA_CONST_3;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        working_state[i] = state[i];
    }

    #pragma unroll
    for (int i = 0; i < 10; i++) {
        QUARTER_ROUND(working_state[0], working_state[4], working_state[8], working_state[12]);
        QUARTER_ROUND(working_state[5], working_state[9], working_state[13], working_state[1]);
        QUARTER_ROUND(working_state[10], working_state[14], working_state[2], working_state[6]);
        QUARTER_ROUND(working_state[15], working_state[3], working_state[7], working_state[11]);

        QUARTER_ROUND(working_state[0], working_state[1], working_state[2], working_state[3]);
        QUARTER_ROUND(working_state[5], working_state[6], working_state[7], working_state[4]);
        QUARTER_ROUND(working_state[10], working_state[11], working_state[8], working_state[9]);
        QUARTER_ROUND(working_state[15], working_state[12], working_state[13], working_state[14]);
    }

    int byte_offset = global_id * RECORD_BYTES;

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t final_word = working_state[i] + state[i];
        uint32_t pt_word = ((const uint32_t*)plaintext)[(byte_offset / 4) + i];
        ((uint32_t*)ciphertext)[(byte_offset / 4) + i] = final_word ^ pt_word;
    }
}

// ==========================================
// KYBER-512 KERNELS (register-heavy, LOW thread count per block)
// --------------------------------------------------------------------------
// WHY THESE ARE SEPARATE KERNELS FROM THE SALSA20 PASS (important — this is a
// fix for a real launch failure, not a stylistic choice):
//
// CUDA allocates ONE register count per thread, uniformly across every thread
// in a block — it cannot give one thread more registers than its neighbor.
// The Kyber keygen/encaps/decaps code path (NTTs, matrix generation, Keccak)
// needs a large number of registers/local-memory slots. If that code is
// inlined into a kernel launched with THREADS_PER_BLOCK=512 (even if only 1
// of every 128 threads actually executes it), the compiler must reserve that
// same large register budget for ALL 512 threads in the block. registers/
// thread * 512 then exceeds the SM's register file, and the launch fails with
// exactly the "too many resources requested for launch" error.
//
// Fix: run Kyber with a SMALL block size (one thread = one full session, no
// tiling needed), and store its output (the derived 256-bit session key) in
// a plain global-memory array. The separate, much lighter Salsa20 kernels
// then read that key from global memory — still zero host transfer, just no
// longer sharing a register budget with the heavy KEM code.
// ==========================================
#define KYBER_BLOCK_SIZE 64   // small on purpose — leaves ample registers/thread

static void kyber_keygen_kernel(
    const uint8_t* seeds,
    uint8_t* pk_out,
    uint8_t* sk_out,
    int num_sessions)
{
    for (int s = 0; s < num_sessions; s++) {
        kyb_kem_keygen(seeds + (size_t)s*32,
                       pk_out + (size_t)s*KYB_PK_BYTES,
                       sk_out + (size_t)s*KYB_SK_BYTES);
    }
}

static void kyber_encaps_kernel(
    const uint8_t* kyber_pk,
    const uint8_t* encaps_msgs,
    uint8_t* kyber_ct_out,
    uint32_t* session_keys_out,
    int num_sessions)
{
    for (int s = 0; s < num_sessions; s++) {
        uint8_t ss[32];
        kyb_kem_encaps(kyber_pk + (size_t)s*KYB_PK_BYTES,
                       encaps_msgs + (size_t)s*32,
                       kyber_ct_out + (size_t)s*KYB_CT_BYTES,
                       ss);
        for (int i = 0; i < KEY_WORDS; i++) {
            session_keys_out[s*KEY_WORDS + i] =
                ((uint32_t)ss[4*i+0]) | ((uint32_t)ss[4*i+1] << 8) |
                ((uint32_t)ss[4*i+2] << 16) | ((uint32_t)ss[4*i+3] << 24);
        }
    }
}

static void kyber_decaps_kernel(
    const uint8_t* kyber_sk,
    const uint8_t* kyber_ct,
    uint32_t* session_keys_out,
    int num_sessions)
{
    for (int s = 0; s < num_sessions; s++) {
        uint8_t ss[32];
        kyb_kem_decaps(kyber_sk + (size_t)s*KYB_SK_BYTES,
                       kyber_ct + (size_t)s*KYB_CT_BYTES,
                       ss);
        for (int i = 0; i < KEY_WORDS; i++) {
            session_keys_out[s*KEY_WORDS + i] =
                ((uint32_t)ss[4*i+0]) | ((uint32_t)ss[4*i+1] << 8) |
                ((uint32_t)ss[4*i+2] << 16) | ((uint32_t)ss[4*i+3] << 24);
        }
    }
}

// ==========================================
// SALSA20 KERNELS (register-light, HIGH thread count per block)
// Pure symmetric-cipher kernels now — no Kyber code inlined here at all, so
// THREADS_PER_BLOCK=512 is safe. Each thread reads its session's already-
// derived key from global memory (session_keys), written by the Kyber
// kernels above. Consecutive threads within a KEM_TILE_SIZE-record group
// read the same address, so this is an L1/L2-cache-friendly broadcast in
// practice, not a real bandwidth cost.
// ==========================================
static void batched_hybrid_crypto_kernel(
    const uint8_t* plaintext_records,
    uint8_t* ciphertext_records,
    const uint32_t* session_keys,
    uint64_t nonce,
    int num_records)
{
    for (int global_record_id = 0; global_record_id < num_records; global_record_id++) {
        int session = global_record_id / KEM_TILE_SIZE;
        uint32_t local_key[KEY_WORDS];
        for (int i = 0; i < KEY_WORDS; i++)
            local_key[i] = session_keys[session*KEY_WORDS + i];

        salsa20_block_process(local_key, nonce, global_record_id,
                              plaintext_records, ciphertext_records, global_record_id);

        for (int i = 0; i < KEY_WORDS; i++) local_key[i] = 0;
    }
}

// ==========================================
// DECRYPT + VERIFY KERNEL: one thread per record
// Salsa20 is its own inverse under XOR, so decryption regenerates the same
// keystream and XORs it against the ciphertext. After decrypting, the record's
// checksum is recomputed and compared against the checksum embedded at encrypt
// time. The Kyber shared secret was already recovered by kyber_decaps_kernel
// into session_keys — this kernel stays pure-Salsa20 for the same register-
// pressure reason as the encrypt kernel above.
// ==========================================
static void batched_hybrid_decrypt_verify_kernel(
    const uint8_t* ciphertext_records,
    uint8_t* decrypted_records,
    uint8_t* verify_flags,
    const uint32_t* session_keys,
    uint64_t nonce,
    int num_records)
{
    for (int global_record_id = 0; global_record_id < num_records; global_record_id++) {
        int session = global_record_id / KEM_TILE_SIZE;
        uint32_t local_key[KEY_WORDS];
        for (int i = 0; i < KEY_WORDS; i++)
            local_key[i] = session_keys[session*KEY_WORDS + i];

        salsa20_block_process(local_key, nonce, global_record_id,
                              ciphertext_records, decrypted_records, global_record_id);

        uint8_t* rec = decrypted_records + (size_t)global_record_id * RECORD_BYTES;
        uint32_t recomputed = fnv1a_32(rec, 48);
        uint32_t stored;
        memcpy(&stored, rec + 48, sizeof(uint32_t));
        verify_flags[global_record_id] = (recomputed == stored) ? 1 : 0;

        for (int i = 0; i < KEY_WORDS; i++) local_key[i] = 0;
    }
}

// ==========================================
// HOST: parse the Intel Berkeley Lab sensor dataset
// Expected whitespace-separated columns:
//   date time epoch moteid temperature humidity light voltage
// Malformed / incomplete lines are skipped.
// ==========================================
static int count_valid_rows(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    char line[512];
    int count = 0;
    SensorRow tmp;
    while (fgets(line, sizeof(line), f)) {
        int n = sscanf(line, "%10s %12s %d %d %f %f %f %f",
                        tmp.date, tmp.time_str, &tmp.epoch, &tmp.moteid,
                        &tmp.temperature, &tmp.humidity, &tmp.light, &tmp.voltage);
        if (n == 8) count++;
    }
    fclose(f);
    return count;
}

static int parse_rows(const char* path, SensorRow* rows, int max_rows) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;

    char line[512];
    int count = 0;
    while (fgets(line, sizeof(line), f) && count < max_rows) {
        SensorRow r;
        int n = sscanf(line, "%10s %12s %d %d %f %f %f %f",
                        r.date, r.time_str, &r.epoch, &r.moteid,
                        &r.temperature, &r.humidity, &r.light, &r.voltage);
        if (n == 8) {
            rows[count] = r;
            count++;
        }
    }
    fclose(f);
    return count;
}

static void pack_record(const SensorRow* row, PackedRecord* out) {
    memset(out, 0, sizeof(PackedRecord));
    out->epoch = row->epoch;
    out->moteid = row->moteid;
    out->temperature = row->temperature;
    out->humidity = row->humidity;
    out->light = row->light;
    out->voltage = row->voltage;
    strncpy(out->date, row->date, sizeof(out->date) - 1);
    strncpy(out->time_str, row->time_str, sizeof(out->time_str) - 1);
    out->checksum = fnv1a_32((const uint8_t*)out, 48);
}


// ==========================================
// MAIN
// CPU VERSION: same pipeline, timing stages, and throughput calculations
// as the CUDA version. CUDA events/allocations/transfers are replaced by
// CPU wall-clock timing and ordinary host memory.
// ==========================================
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int fill_random_bytes(uint8_t* dst, size_t n) {
    FILE* f = fopen("/dev/urandom", "rb");
    if (f) {
        size_t got = fread(dst, 1, n, f);
        fclose(f);
        if (got == n) return 1;
    }

    /* Fallback only if /dev/urandom is unavailable. */
    static int seeded = 0;
    if (!seeded) {
        seeded = 1;
        srand((unsigned)time(NULL));
    }
    for (size_t i = 0; i < n; i++)
        dst[i] = (uint8_t)(rand() & 0xFF);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <path_to_intel_lab_data.txt> [max_records]\n", argv[0]);
        return 1;
    }
    const char* dataset_path = argv[1];
    int record_cap = (argc >= 3) ? atoi(argv[2]) : 2000000; // safety cap for the ~2.3M-row dataset

    int available = count_valid_rows(dataset_path);
    if (available < 0) {
        printf("Could not open dataset file: %s\n", dataset_path);
        return 1;
    }
    int num_records = (available < record_cap) ? available : record_cap;
    if (num_records <= 0) {
        printf("No valid rows found in %s\n", dataset_path);
        return 1;
    }
    printf("Found %d valid rows, processing %d.\n", available, num_records);

    SensorRow* rows = (SensorRow*)malloc(sizeof(SensorRow) * (size_t)num_records);
    if (!rows) {
        printf("Failed to allocate SensorRow buffer.\n");
        return 1;
    }

    int parsed = parse_rows(dataset_path, rows, num_records);
    if (parsed != num_records) {
        printf("Warning: expected %d rows, parsed %d.\n", num_records, parsed);
        num_records = parsed;
    }
    if (num_records <= 0) {
        free(rows);
        return 1;
    }

    // Pad up to a multiple of THREADS_PER_BLOCK with zeroed dummy records so the
    // grid divides evenly; padding records are ignored on the way back out.
    int padded_records = ((num_records + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) * THREADS_PER_BLOCK;
    size_t payload_size = (size_t)padded_records * RECORD_BYTES;

    // ==========================================================================
    // CPU MEMORY DESIGN
    // --------------------------------------------------------------------------
    // The CUDA device buffers are represented by ordinary CPU buffers. The same
    // logical pipeline is retained:
    //   plaintext -> Kyber keygen -> Kyber encaps -> Salsa20 encrypt
    //   -> Kyber decaps -> Salsa20 decrypt+verify -> ciphertext output.
    // No CUDA transfer is performed because all buffers are CPU-resident.
    // ==========================================================================

    PackedRecord* h_plaintext = (PackedRecord*)calloc((size_t)padded_records, sizeof(PackedRecord));
    uint8_t* h_ciphertext = (uint8_t*)malloc(payload_size);
    uint8_t* h_verify = (uint8_t*)calloc((size_t)padded_records, 1);
    PackedRecord* h_sample = (PackedRecord*)calloc(3, sizeof(PackedRecord));

    if (!h_plaintext || !h_ciphertext || !h_verify || !h_sample) {
        printf("Failed to allocate payload buffers.\n");
        free(rows);
        free(h_plaintext);
        free(h_ciphertext);
        free(h_verify);
        free(h_sample);
        return 1;
    }

    for (int i = 0; i < num_records; i++) {
        pack_record(&rows[i], &h_plaintext[i]);
    }
    free(rows);

    uint8_t *d_plaintext = (uint8_t*)h_plaintext;
    uint8_t *d_ciphertext = h_ciphertext;
    uint8_t *d_decrypted = (uint8_t*)calloc((size_t)padded_records, RECORD_BYTES);
    uint8_t *d_verify = h_verify;

    if (!d_decrypted) {
        printf("Failed to allocate decrypted buffer.\n");
        free(h_plaintext);
        free(h_ciphertext);
        free(h_verify);
        free(h_sample);
        return 1;
    }

    uint64_t nonce = 123456789ULL; // demo nonce; use a fresh random nonce per run in practice
    int grid_blocks = padded_records / THREADS_PER_BLOCK;
    int num_sessions = grid_blocks * SESSIONS_PER_BLOCK; // one Kyber-512 session per KEM_TILE_SIZE records

    // ---------------- Kyber-512 keypair + per-session randomness setup ----------------
    // Seeds for keygen and the encapsulation message use the same 32-byte/session
    // logical inputs as the CUDA implementation.
    uint8_t* h_seeds = (uint8_t*)malloc((size_t)num_sessions * 32);
    uint8_t* h_msgs  = (uint8_t*)malloc((size_t)num_sessions * 32);

    uint8_t *d_seeds = h_seeds;
    uint8_t *d_msgs = h_msgs;
    uint8_t *d_kyber_pk = (uint8_t*)malloc((size_t)num_sessions * KYB_PK_BYTES);
    uint8_t *d_kyber_sk = (uint8_t*)malloc((size_t)num_sessions * KYB_SK_BYTES);
    uint8_t *d_kyber_ct = (uint8_t*)malloc((size_t)num_sessions * KYB_CT_BYTES);
    uint32_t *d_session_keys_enc =
        (uint32_t*)malloc((size_t)num_sessions * KEY_WORDS * sizeof(uint32_t));
    uint32_t *d_session_keys_dec =
        (uint32_t*)malloc((size_t)num_sessions * KEY_WORDS * sizeof(uint32_t));

    if (!h_seeds || !h_msgs || !d_kyber_pk || !d_kyber_sk || !d_kyber_ct ||
        !d_session_keys_enc || !d_session_keys_dec) {
        printf("Failed to allocate Kyber/session buffers.\n");
        free(h_plaintext);
        free(h_ciphertext);
        free(h_verify);
        free(h_sample);
        free(d_decrypted);
        free(h_seeds);
        free(h_msgs);
        free(d_kyber_pk);
        free(d_kyber_sk);
        free(d_kyber_ct);
        free(d_session_keys_enc);
        free(d_session_keys_dec);
        return 1;
    }

    fill_random_bytes(h_seeds, (size_t)num_sessions * 32);
    fill_random_bytes(h_msgs,  (size_t)num_sessions * 32);

    printf("Generating %d Kyber-512 session keypairs (pk=%dB, sk=%dB, ct=%dB each)...\n",
           num_sessions, KYB_PK_BYTES, KYB_SK_BYTES, KYB_CT_BYTES);

    // Same logical grid calculation as the CUDA version.
    int kyber_grid = (num_sessions + KYBER_BLOCK_SIZE - 1) / KYBER_BLOCK_SIZE;
    (void)kyber_grid;

    // ---------------- Per-stage timing ----------------
    // CPU wall-clock timing replaces cudaEventElapsedTime. The stage boundaries
    // and reported throughput formulas are kept unchanged.
    double t_start, t_keygen, t_encaps, t_encrypt, t_decaps, t_decrypt, t_d2h_done;

    t_start = now_ms();

    kyber_keygen_kernel(
        d_seeds, d_kyber_pk, d_kyber_sk, num_sessions);
    t_keygen = now_ms();

    kyber_encaps_kernel(
        d_kyber_pk, d_msgs, d_kyber_ct, d_session_keys_enc, num_sessions);
    t_encaps = now_ms();

    printf("Encrypting %d records (%d padded) with %d blocks of %d threads...\n",
           num_records, padded_records, grid_blocks, THREADS_PER_BLOCK);

    batched_hybrid_crypto_kernel(
        d_plaintext, d_ciphertext, d_session_keys_enc, nonce, padded_records);
    t_encrypt = now_ms();

    kyber_decaps_kernel(
        d_kyber_sk, d_kyber_ct, d_session_keys_dec, num_sessions);
    t_decaps = now_ms();

    batched_hybrid_decrypt_verify_kernel(
        d_ciphertext, d_decrypted, d_verify, d_session_keys_dec, nonce, padded_records);
    t_decrypt = now_ms();

    // CPU version has no device->host copy; preserve the same timing stage.
    memcpy(h_sample, d_decrypted, sizeof(PackedRecord) * 3);
    t_d2h_done = now_ms();

    double ms_keygen  = t_keygen  - t_start;
    double ms_encaps  = t_encaps  - t_keygen;
    double ms_encrypt = t_encrypt - t_encaps;
    double ms_decaps  = t_decaps  - t_encrypt;
    double ms_decrypt = t_decrypt - t_decaps;
    double ms_d2h     = t_d2h_done - t_decrypt;
    double ms_total   = t_d2h_done - t_start;

    double mb_payload = payload_size / (1024.0 * 1024.0);
    double mb_useful  = ((double)num_records * RECORD_BYTES) / (1024.0 * 1024.0);

    double encrypt_s      = ms_encrypt / 1000.0 + 1e-9;
    double decrypt_s      = ms_decrypt / 1000.0 + 1e-9;
    double full_pipeline_s = (ms_keygen + ms_encaps + ms_encrypt + ms_decaps + ms_decrypt) / 1000.0 + 1e-9;

    printf("\n---------------- Timing (GPU-clock, ms) ----------------\n");
    printf("  Kyber keygen   (%6d sessions): %9.3f ms  (%9.1f sessions/s)\n",
           num_sessions, ms_keygen,  num_sessions / (ms_keygen/1000.0 + 1e-9));
    printf("  Kyber encaps   (%6d sessions): %9.3f ms  (%9.1f sessions/s)\n",
           num_sessions, ms_encaps,  num_sessions / (ms_encaps/1000.0 + 1e-9));
    printf("  Salsa20 encrypt(%6d records ): %9.3f ms  (%9.1f rec/s, %7.2f MB/s)\n",
           num_records, ms_encrypt, num_records / encrypt_s, mb_payload / encrypt_s);
    printf("  Kyber decaps   (%6d sessions): %9.3f ms  (%9.1f sessions/s)\n",
           num_sessions, ms_decaps,  num_sessions / (ms_decaps/1000.0 + 1e-9));
    printf("  Salsa20 decrypt+verify        : %9.3f ms  (%9.1f rec/s, %7.2f MB/s)\n",
           ms_decrypt, num_records / decrypt_s, mb_payload / decrypt_s);
    printf("  Device->Host copies (ct+verify+sample): %9.3f ms\n", ms_d2h);
    printf("  ---------------------------------------------------------\n");
    printf("  TOTAL (keygen..d2h)           : %9.3f ms\n", ms_total);
    printf("----------------------------------------------------------\n\n");

    printf("================= THROUGHPUT SUMMARY =================\n");
    printf("  Encrypt only  : %10.1f records/s   |  %8.2f MB/s (padded)  |  %8.2f MB/s (useful)\n",
           num_records / encrypt_s, mb_payload / encrypt_s, mb_useful / encrypt_s);
    printf("  Decrypt only  : %10.1f records/s   |  %8.2f MB/s (padded)  |  %8.2f MB/s (useful)\n",
           num_records / decrypt_s, mb_payload / decrypt_s, mb_useful / decrypt_s);
    printf("  Full pipeline : %10.1f records/s   |  %8.2f MB/s (padded)  |  %8.2f MB/s (useful)\n",
           num_records / full_pipeline_s, mb_payload / full_pipeline_s, mb_useful / full_pipeline_s);
    printf("  (Full pipeline = keygen + encaps + encrypt + decaps + decrypt; excludes\n");
    printf("   the initial host->device dataset load and the final device->host copies,\n");
    printf("   consistent with how GPU-cipher throughput is conventionally reported.)\n");
    printf("========================================================\n\n");

    // Write ciphertext to disk with the same header.
    FILE* out = fopen("sensor_ciphertext.bin", "wb");
    if (out) {
        const char magic[4] = {'S','C','R','1'};
        fwrite(magic, 1, 4, out);
        fwrite(&num_records, sizeof(int), 1, out);
        fwrite(&nonce, sizeof(uint64_t), 1, out);
        fwrite(h_ciphertext, 1, payload_size, out);
        fclose(out);
        printf("Wrote ciphertext to sensor_ciphertext.bin\n");
    }

    int verified_ok = 0;
    for (int i = 0; i < num_records; i++) {
        if (h_verify[i]) verified_ok++;
    }
    printf("Verification: %d / %d records passed integrity check.\n", verified_ok, num_records);

    for (int i = 0; i < num_records && i < 3; i++) {
        printf("  [%d] %s %s epoch=%d moteid=%d temp=%.4f hum=%.4f light=%.2f volt=%.5f verify=%s\n",
               i, h_sample[i].date, h_sample[i].time_str, h_sample[i].epoch, h_sample[i].moteid,
               h_sample[i].temperature, h_sample[i].humidity, h_sample[i].light, h_sample[i].voltage,
               h_verify[i] ? "OK" : "FAILED");
    }

    free(d_plaintext);
    free(d_ciphertext);
    free(d_decrypted);
    free(d_verify);
    free(d_seeds);
    free(d_msgs);
    free(d_kyber_pk);
    free(d_kyber_sk);
    free(d_kyber_ct);
    free(d_session_keys_enc);
    free(d_session_keys_dec);
    free(h_sample);

    return 0;
}
