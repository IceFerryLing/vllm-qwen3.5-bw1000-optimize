#if !defined(__DU_MMA_H__)
#define __DU_MMA_H__

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__)  || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
#else
#error "Only support gfx926 gfx928 gfx936 gfx938"
#endif

#define __DU_MMA_DEVICE_DECL__ static __device__ __inline__

#if defined(__cplusplus) && defined(__HIPCC__)

namespace du
{
namespace dumma
{

  __host__ __device__ float __du_float_to_tf32(float in)
  {
      union
      {
          float fp32;
          uint32_t int32;
      } u = {in};
      if (~u.int32 & 0x7f800000)
      {

          u.int32 += 0xfff + ((u.int32 >> 13) & 1);
      }
      else if (u.int32 & 0x1fff)
      {

          u.int32 |= 0x2000;
      }

      u.int32 &= 0xffffe000;
      return u.fp32;
  }

struct row_major;
struct col_major;
struct matrix_a;
struct matrix_b;
struct accumulator;

enum layout_t
{
    mem_row_major,
    mem_col_major
};

namespace precision
{
    struct tf32;
}




namespace experimental
{
    namespace precision
    {
        struct s4;
        struct u4;
        struct b1;
    }
}

template <typename T>
struct helper_traits {
    typedef T element_type;
    typedef T storage_element_type;
    typedef T fill_argument_type;
};

template <>
struct helper_traits<precision::tf32>
{
    typedef precision::tf32 element_type;
    typedef float storage_element_type;
    typedef float fill_argument_type;
};

template <>
struct helper_traits<experimental::precision::u4>
{
    typedef experimental::precision::u4 element_type;
    typedef int storage_element_type;
    typedef int fill_argument_type;
};

template <>
struct helper_traits<experimental::precision::s4>
{
    typedef experimental::precision::s4 element_type;
    typedef int storage_element_type;
    typedef int fill_argument_type;
};

template <>
struct helper_traits<experimental::precision::b1>
{
    typedef experimental::precision::b1 element_type;
    typedef unsigned storage_element_type;
    typedef unsigned fill_argument_type;
};



template <typename T, int size, int packed_size = size>
struct __align__(8) DUFragmentBase
{

    
    enum
    {
        num_elements = size
    };

    enum
    {
        num_storage_elements = packed_size
    };

    
    typedef T element_type;
    typedef typename helper_traits<T>::storage_element_type storage_element_type;

    storage_element_type x[num_storage_elements] = {};
};

template <typename FragEleType, typename StorageType, typename ArgType>
static inline __device__ StorageType __du_get_storage_value(ArgType in) { return in; }

template <>
__device__ inline int
__du_get_storage_value<experimental::precision::u4, int, int>(int in)
{
  
  int val = in & 0xf;
  return (val | (val << 4) | (val << 8) | (val << 12) |
          (val << 16) | (val << 20) | (val << 24) | (val << 28));

      
};

template<>
__device__ inline int
__du_get_storage_value<experimental::precision::s4, int, int>(int in)
{
  
  int val = in & 0xf;
  return (val | (val << 4) | (val << 8) | (val << 12) |
          (val << 16) | (val << 20) | (val << 24) | (val << 28));
};

template<>
__device__ inline unsigned 
__du_get_storage_value<experimental::precision::b1, unsigned, unsigned>(unsigned in)
{

return (in & 0x1) ? 0xFFFFFFFFU : 0;
}

template <typename FragEleType, int size, int packed_size>
__DU_MMA_DEVICE_DECL__ void du_fill_fragment(DUFragmentBase<FragEleType, size, packed_size>& f, 
            
const typename helper_traits<FragEleType>::fill_argument_type & in) {


typedef typename helper_traits<FragEleType>::storage_element_type storage_type;
storage_type v = __du_get_storage_value<FragEleType, storage_type>(in);
#pragma unroll
for (int i = 0; i < f.num_storage_elements; i++)
    f.x[i] = v;
}


template<typename Use, int m, int n, int k, typename T, typename Layout=void> class DUFragment;




#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__)  || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
template<> class DUFragment<matrix_a, 16, 16, 4, float, row_major> : public DUFragmentBase<float, 1> {};
template<> class DUFragment<matrix_a, 16, 16, 4, float, col_major> : public DUFragmentBase<float, 1> {};
template<> class DUFragment<matrix_b, 16, 16, 4, float, row_major> : public DUFragmentBase<float, 1> {};
template<> class DUFragment<matrix_b, 16, 16, 4, float, col_major> : public DUFragmentBase<float, 1> {};
template<> class DUFragment<accumulator, 16, 16, 4, float> : public DUFragmentBase<float, 4> {};
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx936__) || defined(__gfx938__)
template<> class DUFragment<matrix_a, 16, 16, 4, double, row_major> : public DUFragmentBase<double, 1> {};
template<> class DUFragment<matrix_a, 16, 16, 4, double, col_major> : public DUFragmentBase<double, 1> {};
template<> class DUFragment<matrix_b, 16, 16, 4, double, row_major> : public DUFragmentBase<double, 1> {};
template<> class DUFragment<matrix_b, 16, 16, 4, double, col_major> : public DUFragmentBase<double, 1> {};
template<> class DUFragment<accumulator, 16, 16, 4, double> : public DUFragmentBase<double, 4> {};
#endif




#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__) 
template<> class DUFragment<matrix_a, 16, 16, 8, float, row_major> : public DUFragmentBase<float, 2> {};
template<> class DUFragment<matrix_a, 16, 16, 8, float, col_major> : public DUFragmentBase<float, 2> {};
template<> class DUFragment<matrix_b, 16, 16, 8, float, row_major> : public DUFragmentBase<float, 2> {};
template<> class DUFragment<matrix_b, 16, 16, 8, float, col_major> : public DUFragmentBase<float, 2> {};

template<> class DUFragment<matrix_a, 16, 16, 8, precision::tf32, row_major> : public DUFragmentBase<precision::tf32, 2> {};
template<> class DUFragment<matrix_a, 16, 16, 8, precision::tf32, col_major> : public DUFragmentBase<precision::tf32, 2> {};
template<> class DUFragment<matrix_b, 16, 16, 8, precision::tf32, row_major> : public DUFragmentBase<precision::tf32, 2> {};
template<> class DUFragment<matrix_b, 16, 16, 8, precision::tf32, col_major> : public DUFragmentBase<precision::tf32, 2> {};

template<> class DUFragment<accumulator, 16, 16, 8, float> : public DUFragmentBase<float, 4> {};




template<> class DUFragment<matrix_a, 16, 16, 16, __half, row_major> : public DUFragmentBase<__half, 4> {};
template<> class DUFragment<matrix_a, 16, 16, 16, __half, col_major> : public DUFragmentBase<__half, 4> {};
template<> class DUFragment<matrix_b, 16, 16, 16, __half, row_major> : public DUFragmentBase<__half, 4> {};
template<> class DUFragment<matrix_b, 16, 16, 16, __half, col_major> : public DUFragmentBase<__half, 4> {};

template<> class DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, row_major> : public DUFragmentBase<__hip_bfloat16, 4> {};
template<> class DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, col_major> : public DUFragmentBase<__hip_bfloat16, 4> {};
template<> class DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, row_major> : public DUFragmentBase<__hip_bfloat16, 4> {};
template<> class DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, col_major> : public DUFragmentBase<__hip_bfloat16, 4> {};

template<> class DUFragment<accumulator, 16, 16, 16, float> : public DUFragmentBase<float, 4> {};




template<> class DUFragment<matrix_a, 16, 16, 32, signed char, row_major> : public DUFragmentBase<signed char, 8> {};
template<> class DUFragment<matrix_a, 16, 16, 32, signed char, col_major> : public DUFragmentBase<signed char, 8> {};
template<> class DUFragment<matrix_b, 16, 16, 32, signed char, row_major> : public DUFragmentBase<signed char, 8> {};
template<> class DUFragment<matrix_b, 16, 16, 32, signed char, col_major> : public DUFragmentBase<signed char, 8> {};

template<> class DUFragment<matrix_a, 16, 16, 32, unsigned char, row_major> : public DUFragmentBase<unsigned char, 8> {};
template<> class DUFragment<matrix_a, 16, 16, 32, unsigned char, col_major> : public DUFragmentBase<unsigned char, 8> {};
template<> class DUFragment<matrix_b, 16, 16, 32, unsigned char, row_major> : public DUFragmentBase<unsigned char, 8> {};
template<> class DUFragment<matrix_b, 16, 16, 32, unsigned char, col_major> : public DUFragmentBase<unsigned char, 8> {};

template<> class DUFragment<accumulator, 16, 16, 32, int> : public DUFragmentBase<int, 4> {};
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__) 
template<> class DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, row_major> : public DUFragmentBase<__hip_fp8_e4m3, 8> {};
template<> class DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, col_major> : public DUFragmentBase<__hip_fp8_e4m3, 8> {};
template<> class DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, row_major> : public DUFragmentBase<__hip_fp8_e4m3, 8> {};
template<> class DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, col_major> : public DUFragmentBase<__hip_fp8_e4m3, 8> {};

template<> class DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, row_major> : public DUFragmentBase<__hip_fp8_e5m2, 8> {};
template<> class DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, col_major> : public DUFragmentBase<__hip_fp8_e5m2, 8> {};
template<> class DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, row_major> : public DUFragmentBase<__hip_fp8_e5m2, 8> {};
template<> class DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, col_major> : public DUFragmentBase<__hip_fp8_e5m2, 8> {};
template<> class DUFragment<accumulator, 16, 16, 32, float> : public DUFragmentBase<float, 4> {};
#endif



#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__) 
template<> class DUFragment<matrix_a, 16, 16, 64, experimental::precision::s4, row_major> : public DUFragmentBase<experimental::precision::s4, 2> {};
template<> class DUFragment<matrix_b, 16, 16, 64, experimental::precision::s4, col_major> : public DUFragmentBase<experimental::precision::s4, 2> {};

template<> class DUFragment<matrix_a, 16, 16, 64, experimental::precision::u4, row_major> : public DUFragmentBase<experimental::precision::u4, 2> {};
template<> class DUFragment<matrix_b, 16, 16, 64, experimental::precision::u4, col_major> : public DUFragmentBase<experimental::precision::u4, 2> {};

template<> class DUFragment<accumulator, 16, 16, 64, int> : public DUFragmentBase<int, 4> {};
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__)  || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 4, float, row_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 4, float, col_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 4, float, row_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 4, float, col_major>& a, const float* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 4, float>& a, const float* p, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__)  || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 4, double, row_major>& a, const double* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 4, double, col_major>& a, const double* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 4, double, row_major>& a, const double* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 4, double, col_major>& a, const double* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 4, double>& a, const double* p, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 8, float, row_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 8, float, col_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 8, float, row_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 8, float, col_major>& a, const float* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 8, precision::tf32, row_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 8, precision::tf32, col_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 8, precision::tf32, row_major>& a, const float* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 8, precision::tf32, col_major>& a, const float* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 8, float>& a, const float* p, unsigned ldm, layout_t layout);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 16, __half, row_major>& a, const __half* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 16, __half, col_major>& a, const __half* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 16, __half, row_major>& a, const __half* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 16, __half, col_major>& a, const __half* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, row_major>& a, const __hip_bfloat16* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, col_major>& a, const __hip_bfloat16* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, row_major>& a, const __hip_bfloat16* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, col_major>& a, const __hip_bfloat16* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 16, float>& a, const float* p, unsigned ldm, layout_t layout);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, signed char, row_major>& a, const signed char* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, signed char, col_major>& a, const signed char* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, signed char, row_major>& a, const signed char* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, signed char, col_major>& a, const signed char* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, unsigned char, row_major>& a, const unsigned char* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, unsigned char, col_major>& a, const unsigned char* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 32, int>& a, const int* p, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, row_major>& a, const __hip_fp8_e4m3* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, col_major>& a, const __hip_fp8_e4m3* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, row_major>& a, const __hip_fp8_e4m3* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, col_major>& a, const __hip_fp8_e4m3* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, row_major>& a, const __hip_fp8_e5m2* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, col_major>& a, const __hip_fp8_e5m2* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, row_major>& a, const __hip_fp8_e5m2* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, col_major>& a, const __hip_fp8_e5m2* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 32, float>& a, const float* p, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 64, experimental::precision::s4, row_major>& a, const void* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 64, experimental::precision::s4, col_major>& a, const void* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_a, 16, 16, 64, experimental::precision::u4, row_major>& a, const void* p, unsigned ldm);
__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<matrix_b, 16, 16, 64, experimental::precision::u4, col_major>& a, const void* p, unsigned ldm);

__DU_MMA_DEVICE_DECL__ void du_load_matrix_sync(DUFragment<accumulator, 16, 16, 64, int>& a, const int* p, unsigned ldm, layout_t layout);

#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__)  || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(float *p, const DUFragment<accumulator, 16, 16, 4, float>& a, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(double *p, const DUFragment<accumulator, 16, 16, 4, double>& a, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(float *p, const DUFragment<accumulator, 16, 16, 8, float>& a, unsigned ldm, layout_t layout);
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(float *p, const DUFragment<accumulator, 16, 16, 16, float>& a, unsigned ldm, layout_t layout);
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(int *p, const DUFragment<accumulator, 16, 16, 32, int>& a, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(float *p, const DUFragment<accumulator, 16, 16, 32, float>& a, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
__DU_MMA_DEVICE_DECL__ void du_store_matrix_sync(int *p, const DUFragment<accumulator, 16, 16, 64, int>& a, unsigned ldm, layout_t layout);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__)  || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 4, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 4, float, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 4, float, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 4, float>& c);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx926__) || defined(__gfx936__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 4, double>& d,
                                        const DUFragment<matrix_a, 16, 16, 4, double, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 4, double, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 4, double>& c);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx928__) || defined(__gfx936__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 8, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 8, float, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 8, float, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 8, float>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 8, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 8, precision::tf32, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 8, precision::tf32, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 8, float>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 16, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 16, __half, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 16, __half, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 16, float>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 16, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 16, __hip_bfloat16, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 16, __hip_bfloat16, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 16, float>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, int>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, signed char, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, signed char, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, int>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, int>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, unsigned char, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, unsigned char, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, int>& c);
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, float>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, float>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e4m3, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e5m2, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, float>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 32, float>& d,
                                        const DUFragment<matrix_a, 16, 16, 32, __hip_fp8_e5m2, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 32, __hip_fp8_e4m3, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 32, float>& c);                                      
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx936__) || defined(__gfx938__)
template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 64, int>& d,
                                        const DUFragment<matrix_a, 16, 16, 64, experimental::precision::s4, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 64, experimental::precision::s4, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 64, int>& c);

template <typename LayoutA, typename LayoutB>
__DU_MMA_DEVICE_DECL__ void du_mma_sync(DUFragment<accumulator, 16, 16, 64, int>& d,
                                        const DUFragment<matrix_a, 16, 16, 64, experimental::precision::u4, LayoutA>& a, 
                                        const DUFragment<matrix_b, 16, 16, 64, experimental::precision::u4, LayoutB>& b,
                                        const DUFragment<accumulator, 16, 16, 64, int>& c);
#endif

}
}

#endif 

#undef __DU_MMA_DEVICE_DECL__

#if defined(__HIP_DEVICE_COMPILE__)
 #include "du_mma.hpp"
#endif 

#endif 

