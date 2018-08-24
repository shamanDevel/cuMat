#ifndef __CUMAT_LOGGING_H__
#define __CUMAT_LOGGING_H__

#include "Macros.h"

/**
 * Defines the logging macros.
 * All logging messages are started with a call to
 * e.g. CUMAT_LOG_INFO(message)
 * 
 * You can define the CUMAT_LOG and the related logging levels
 * to point to your own logging implementation.
 * If you don't overwrite these, a very trivial logger is used
 * that simply prints to std::cout.
 * 
 * This is achieved by globally defining CUMAT_LOGGING_PLUGIN
 * that includes a file that then defines all the logging macros:
 * CUMAT_LOG_DEBUG, CUMAT_LOG_INFO, CUMAT_LOG_WARNING, CUMAT_LOG_SEVERE
 */

#ifndef CUMAT_LOG

#include <ostream>
#include <iostream>
#include <string>

#ifdef CUMAT_LOGGING_PLUGIN
#include CUMAT_LOGGING_PLUGIN
#endif

CUMAT_NAMESPACE_BEGIN


class DummyLogger
{
private:
	std::ios_base::fmtflags flags_;
	bool enabled_;

public:
	DummyLogger(const std::string& level)
		: enabled_(level != "[debug]") //to disable the many, many logs during kernel evaluations in the test suites
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

#ifndef CUMAT_LOG_DEBUG
/**
 * Logs the message as a debug message
 */
#define CUMAT_LOG_DEBUG(...) DummyLogger("[debug]") << __VA_ARGS__
#endif

#ifndef CUMAT_LOG_INFO
 /**
 * Logs the message as a information message
 */
#define CUMAT_LOG_INFO(...) DummyLogger("[info]") << __VA_ARGS__
#endif
#ifndef CUMAT_LOG_WARNING
 /**
 * Logs the message as a warning message
 */
#define CUMAT_LOG_WARNING(...) DummyLogger("[warning]") << __VA_ARGS__
#endif
#ifndef CUMAT_LOG_SEVERE
 /**
 * Logs the message as a severe error message
 */
#define CUMAT_LOG_SEVERE(...) DummyLogger("[SEVERE]") << __VA_ARGS__
#endif

CUMAT_NAMESPACE_END

#endif

#endif