/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _DEFINITIONS_DEBUGGING_H_
#define _DEFINITIONS_DEBUGGING_H_

#include <iostream>

// To avoid "controlling expression is constant" warning by nvcc compiler when
// there are messages in the assert function, use this instead of assert.

#define ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                      << " line " << __LINE__ << ": " << message << std::endl; \
            std::terminate(); \
        } \
    } while (false)


#endif  // _DEFINITIONS_DEBUGGING_H_
