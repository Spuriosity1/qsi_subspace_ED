#pragma once


#if __cplusplus >= 202002L
    // C++20
#include <barrier>
using Barrier = std::barrier;

#else

#include <condition_variable>
#include <cstddef>
#include <functional>


// Source - https://stackoverflow.com/a/68181809
// Posted by Yakk - Adam Nevraumont, modified by community. See post 'Timeline' for change history
// Retrieved 2026-01-20, License - CC BY-SA 4.0
struct Barrier {
  mutable std::mutex m;
  std::condition_variable cv;
  std::size_t size;
  std::ptrdiff_t remaining;
  std::ptrdiff_t phase = 0;
  std::function<void()> completion;
  Barrier( std::size_t s, std::function<void()> f ):
    size(s), remaining(s), completion(std::move(f))
  {}
  void arrive_and_wait()
  {
    auto l = std::unique_lock(m);
    --remaining;
    if (remaining != 0)
    {
      auto myphase = phase+1;
      cv.wait(l, [&]{
        return myphase - phase <= 0;
      });
    }
    else
    {
      completion();
      remaining = size;
      ++phase;
      cv.notify_all();
    }
  }
  void arrive_and_drop()
  {
      auto l = std::unique_lock(m);
      --size;
      --remaining;
      if (remaining == 0) {
          completion();
          remaining = size;
          ++phase;
          cv.notify_all();
      }
  }



};
#endif
