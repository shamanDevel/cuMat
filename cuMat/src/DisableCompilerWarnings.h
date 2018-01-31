#ifndef __CUMAT_DISABLE_COMPILER_WARNINGS_H__
#define __CUMAT_DISABLE_COMPILER_WARNINGS_H__


#if defined __GNUC__

//This is a stupid warning in GCC that pops up everytime storage qualifies or so are compared
#pragma GCC diagnostic ignored "-Wenum-compare"

#endif


#endif
