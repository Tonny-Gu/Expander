#pragma once

#include <iostream>
#include <stdio.h>
#include <time.h>
#include <tuple>
#include <cassert>
#include <cstring>

#ifdef __ARM_NEON
#include <arm_neon.h>
#else
#include <immintrin.h>
#endif

#include "basefield.hpp"

namespace gkr::M31_field {

const int mod = 2147483647;
typedef int64 Scalar;
#define mod_reduce_int(x) (x = (((x) & mod) + ((x) >> 31)))

/*
P = 2^31 - 1
    # 2^31 - 1 = 2147483647

(x, y)

(x, y) + (a, b) = ((x + a) mod P, (y + b) mod P)
(x, y) * (a, b) = (x + iy) * (a + ib) = ((x * a - y * b) mod P, (x * b + y * a) mod P)

(x mod P) = (x & P) + (x >> 31)
*/
class M31 final : public BaseField<M31, Scalar>,
                                public FFTFriendlyField<M31>
{
public:
    static M31 INV_2;

    uint32 x;

public:
    static M31 zero() { return new_unchecked(0); }
    static M31 one() { return new_unchecked(1); }
    static std::tuple<Scalar, uint32> size() { return {mod, 2}; }
    static M31 random() { return M31{static_cast<uint32>(rand())}; }  // FIXME: random cannot be used in production

    static inline M31 new_unchecked(uint32 x)
    {
        M31 f;
        f.x = x;
        return f;
    }

public:
    M31() { this->x = 0; }
    M31(uint32 v)
    {
        mod_reduce_int(v);
        this->x = v;
    }

    inline M31 operator+(const M31 &rhs) const
    {
        M31 result;
        result.x = (x + rhs.x);
        if (result.x >= mod)
            result.x -= mod;
        return result;
    }

    inline M31 operator*(const M31 &rhs) const
    {
        int64 xx = static_cast<int64>(x) * rhs.x;

        mod_reduce_int(xx);
        if (xx >= mod)
            xx -= mod;

        return new_unchecked(xx);
    }

    inline M31 operator-() const
    {
        uint32 x = (this->x == 0) ? 0 : (mod - this->x);
        return new_unchecked(x);
    }

    inline M31 operator-(const M31 &rhs) const
    {
        return *this + (-rhs);
    }

    bool operator==(const M31 &rhs) const
    {
        return this->x == rhs.x;
    };

    // using default exp implementation in BaseField without any override

    void to_bytes(uint8* output) const
    {
        memcpy(output, this, sizeof(*this));
    };
    static int byte_length()
    {
        return 4;
    }

    void from_bytes(const uint8* input)
    {
        memcpy(this, input, 4);
        mod_reduce_int(x);
        if (x >= mod) x -= mod;
    };

    friend std::ostream &operator<<(std::ostream &os, const M31 &f)
    {
        os.write((char *)&f.x, sizeof(f.x));
        return os;
    }

    friend std::istream &operator>>(std::istream &is, M31 &f)
    {
        // TODO: FIX INCORRECT READING
        // the input file uses 256 bits to represent a field element
        uint32 repeat = 32 / sizeof(f.x);
        for (uint32 i = 0; i < repeat; i++)
        {
            is.read((char *)&f.x, sizeof(f.x));
        }
        f.x = 1;
        return is;
    }

    static M31 default_rand_sentinel()
    {
        // FIXME: is this a reasonable value?
        return new_unchecked(4294967295 - 1);
    }
};

#ifdef __ARM_NEON
    typedef uint32x4_t DATA_TYPE;
#else
    typedef __m256i DATA_TYPE;
#endif

const int vectorize_size = 64;

class VectorizedM31 final : public BaseField<VectorizedM31, Scalar>,
                                        public FFTFriendlyField<VectorizedM31>
{
    public:
    typedef M31 primitive_type;
    M31 elements[vectorize_size];

    static VectorizedM31 INV_2;
    static VectorizedM31 zero()
    {
        VectorizedM31 z;
        for (int i = 0; i < vectorize_size; i++)
        {
            z.elements[i] = M31::zero();
        }
        return z;
    }
 
    static VectorizedM31 one() {
        return new_unchecked(M31::one());
    }

    static std::tuple<Scalar, uint32> size()
    {
        auto s = mod;
        return {s, 2};
    }

    static VectorizedM31 random()
    {
        VectorizedM31 r;
        for (int i = 0; i < vectorize_size; i++)
        {
            r.elements[i] = M31::random();
        }
        return r;
    }

    static VectorizedM31 random_bool()
    {
        VectorizedM31 r;
        for (int i = 0; i < vectorize_size; i++)
        {
            r.elements[i] = M31(M31::random().x % 2);
        }
        return r;
    }

    inline static VectorizedM31 new_unchecked(const M31 &x)
    {
        VectorizedM31 r;
        for (int i = 0; i < vectorize_size; i++)
        {
            r.elements[i] = x;
        }
        return r;
    }
    
    static VectorizedM31 pack_full(const M31 &f);
    static std::vector<VectorizedM31> pack_field_elements(const std::vector<M31> &fs) {
        uint32 n_seg = (fs.size() + pack_size() - 1) / pack_size(); // ceiling
        std::vector<VectorizedM31> packed_field_elements(n_seg);

        for (uint32 i = 0; i < n_seg - 1; i++)
        {
            for (uint32 j = 0; j < vectorize_size; j++)
            {
                uint32 base = i * pack_size() + j;
                //packed_field_elements[i].elements[j].x = vld1q_u32((uint32_t *)&fs[base]);
                packed_field_elements[i].elements[j].x = fs[base].x;
            }
        }

        // set the remaining value
        uint32 base = (n_seg - 1) * pack_size();
        std::vector<int> x(pack_size());
        auto x_it = x.begin();
        for (uint32 i = base; i < fs.size(); i++)
        {
            *x_it++ = fs[i].x;
        }
        for (uint32 j = 0; j < vectorize_size; j++)
        {
            auto offset = j;
            //packed_field_elements[n_seg - 1].elements[j].x = vld1q_u32((uint32_t *)&x[offset]);
            packed_field_elements[n_seg - 1].elements[j].x = x[offset];
        }

        return packed_field_elements;
    }

public:   

    VectorizedM31() {};

    VectorizedM31(uint32 xx)
    {
        mod_reduce_int(xx);
        for (int i = 0; i < vectorize_size; i++)
        {
            elements[i] = M31(xx);
        }
    }

    VectorizedM31(const VectorizedM31 &x)
    {
        for (int i = 0; i < vectorize_size; i++)
        {
            elements[i] = x.elements[i];
        }
    }

    inline VectorizedM31 operator+(const VectorizedM31 &rhs) const
    {
        VectorizedM31 result;
        for (int i = 0; i < vectorize_size; i++)
        {
            result.elements[i] = elements[i] + rhs.elements[i];
        }
        return result;
    }

    inline VectorizedM31 operator*(const VectorizedM31 &rhs) const
    {
        VectorizedM31 result;
        for (int i = 0; i < vectorize_size; i++)
        {
            result.elements[i] = elements[i] * rhs.elements[i];
        }
        return result;
    }

    inline VectorizedM31 operator*(const M31 &rhs) const
    {
        VectorizedM31 result;
        for (int i = 0; i < vectorize_size; i++)
        {
            result.elements[i] = elements[i] * rhs;
        }
        return result;
    }

    inline VectorizedM31 operator*(const int &rhs) const
    {
        VectorizedM31 result;
        for (int i = 0; i < vectorize_size; i++)
        {
            result.elements[i] = elements[i] * rhs;
        }
        return result;
    }

    inline VectorizedM31 operator+(const M31 &rhs) const
    {
        VectorizedM31 result;
        for (int i = 0; i < vectorize_size; i++)
        {
            result.elements[i] = elements[i] + rhs;
        }
        return result;
    }

    inline VectorizedM31 operator+(const int &rhs) const
    {
        VectorizedM31 result;
        for (int i = 0; i < vectorize_size; i++)
        {
            result.elements[i] = elements[i] + rhs;
        }
        return result;
    }

    inline void operator += (const VectorizedM31 &rhs)
    {
        for (int i = 0; i < vectorize_size; i++)
        {
            elements[i] += rhs.elements[i];
        }
    }

    inline VectorizedM31 operator-() const
    {
        VectorizedM31 r;
        for (int i = 0; i < vectorize_size; i++)
        {
            r.elements[i] = -elements[i];
        }
        return r;
    }

    inline VectorizedM31 operator-(const VectorizedM31 &rhs) const
    {
        VectorizedM31 r;
        for (int i = 0; i < vectorize_size; i++)
        {
            r.elements[i] = elements[i] - rhs.elements[i];
        }
        return r;
    }

    bool operator==(const VectorizedM31 &rhs) const
    {
        for (int i = 0; i < vectorize_size; i++)
        {
            if (!(elements[i] == rhs.elements[i]))
            {
                return false;
            }
        }
        return true;
    }

    void to_bytes(uint8 *output) const
    {
        for (int i = 0; i < vectorize_size; i++)
        {
            elements[i].to_bytes(output + i * sizeof(M31));
        }
    }

    void from_bytes(const uint8 *input)
    {
        for (int i = 0; i < vectorize_size; i++)
        {
            elements[i].from_bytes(input + i * sizeof(M31));
        }
    }

    std::vector<M31> unpack() const {
        std::vector<M31> result;
        for (int i = 0; i < vectorize_size; i++)
        {
            result.push_back(elements[i]);
        }
        return result;
    }

    static size_t pack_size() {
        return vectorize_size;
    }
};

inline M31 M31::INV_2 = (1 << 30);
inline VectorizedM31 VectorizedM31::INV_2 = VectorizedM31::new_unchecked(M31::INV_2);

} // namespace gkr::M31_field