#pragma once

#define EMPTY()
#define DEFER(...) __VA_ARGS__ EMPTY()
#define EVAL(...) __VA_ARGS__
