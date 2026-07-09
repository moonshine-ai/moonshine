/*
 *  Copyright (c) 2003-2010, Mark Borgerding. All rights reserved.
 *  This file is part of KISS FFT - https://github.com/mborgerding/kissfft
 *
 *  SPDX-License-Identifier: BSD-3-Clause
 *  See COPYING file for more information.
 */

#ifndef kiss_fft_log_h
#define kiss_fft_log_h

#define ERROR 1
#define WARNING 2
#define INFO 3
#define DEBUG 4

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

#if defined(NDEBUG)
# define KISS_FFT_LOG_MSG(severity, ...) ((void)0)
#else
# define KISS_FFT_LOG_MSG(severity, ...) \
    fprintf(stderr, "[" #severity "] " __FILE__ ":" TOSTRING(__LINE__) " "); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n")
#endif

#define KISS_FFT_ERROR(...) KISS_FFT_LOG_MSG(ERROR, __VA_ARGS__)
#define KISS_FFT_WARNING(...) KISS_FFT_LOG_MSG(WARNING, __VA_ARGS__)
#define KISS_FFT_INFO(...) KISS_FFT_LOG_MSG(INFO, __VA_ARGS__)
#define KISS_FFT_DEBUG(...) KISS_FFT_LOG_MSG(DEBUG, __VA_ARGS__)



#endif /* kiss_fft_log_h */