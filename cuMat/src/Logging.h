#ifndef __CUMAT_LOGGING_H__
#define __CUMAT_LOGGING_H__

#include "Macros.h"

/**
 * Defines the logging macros.
 * All logging messages are started with a call to CUMAT_LOG(level)
 * where level can be CUMAT_LOG_DEBUG, CUMAT_LOG_INFO,
 * CUMAT_LOG_WARNING or CUMAT_LOG_SEVERE.
 * 
 * You can define the CUMAT_LOG and the related logging levels
 * to point to your own logging implementation.
 * If you don't overwrite these, a very trivial logger is used
 * that simply prints to std::cout.
 */

#ifndef CUMAT_LOG

#include <ostream>
#include <iostream>
#include <string>

CUMAT_NAMESPACE_BEGIN

/**
* The logging level for information messages
*/
#define CUMAT_LOG_DEBUG "[debug]"

/**
* The logging level for information messages
*/
#define CUMAT_LOG_INFO "[info]"

/**
* The logging level for information messages
*/
#define CUMAT_LOG_WARNING "[warning]"

/**
* The logging level for information messages
*/
#define CUMAT_LOG_SEVERE "[SEVERE]"

class DummyLogger
{
private:
	std::ios_base::fmtflags flags_;
	bool enabled_;

public:
	DummyLogger(const std::string& level)
		: enabled_(level != CUMAT_LOG_DEBUG) //to disable the many, many logs during kernel evaluations in the test suites
	{
		flags_ = std::cout.flags();
		if (enabled_) std::cout << level << "  ";
	}
	~DummyLogger()
	{
		if (enabled_) {
			std::cout << std::endl;
			std::cout.flags(flags_);
		}
	}

	template <typename T>
	DummyLogger& operator<<(const T& t) {
		if (enabled_) std::cout << t;
		return *this;
	}
};

/**
 * Returns the logger for the given level.
 */
#define CUMAT_LOG(level) DummyLogger(level)

CUMAT_NAMESPACE_END

#endif

#endif