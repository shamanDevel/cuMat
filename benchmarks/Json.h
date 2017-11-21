#pragma once
#ifndef TOOLS_JSON_JSON_H
#define TOOLS_JSON_JSON_H

#include <string>
#include "json_st.h"

namespace Json
{
	Value ParseFile (const std::string& filename);
	Value ParseString (const std::string& s);
}

#endif