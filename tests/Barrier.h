#ifndef __CUMAT_TESTS_BARRIER_H__
#define __CUMAT_TESTS_BARRIER_H__

#include <mutex>
#include <condition_variable>

//https://stackoverflow.com/a/24465624/4053176
class Barrier
{
private:
	std::mutex _mutex;
	std::condition_variable _cv;
	std::size_t _count;
public:
	explicit Barrier(std::size_t count) : _count{ count } { }
	void Wait()
	{
		std::unique_lock<std::mutex> lock{ _mutex };
		if (--_count == 0) {
			_cv.notify_all();
		}
		else {
			_cv.wait(lock, [this] { return _count == 0; });
		}
	}
};

#endif