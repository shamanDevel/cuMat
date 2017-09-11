#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main()
#include <catch/catch.hpp>
#include <iostream>

struct MyListener : Catch::TestEventListenerBase {

	using TestEventListenerBase::TestEventListenerBase; // inherit constructor

	virtual void testCaseStarting(Catch::TestCaseInfo const& testInfo) override {
		std::cout << "Execute " << testInfo.tagsAsString << " " << testInfo.name << std::endl;
	}
};
CATCH_REGISTER_LISTENER(MyListener)