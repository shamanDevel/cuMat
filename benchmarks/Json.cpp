#include "Json.h"

#include <fstream>
#include <sstream>
#include <string.h>

namespace Json
{
	Value parse_value (std::istringstream& ss);

	Value ParseFile (const std::string& filename)
	{
		std::ifstream file (filename);
		if (!file.is_open ())
		{
			return Json::Value ();
		}
		std::string s ((std::istreambuf_iterator<char> (file)), std::istreambuf_iterator<char> ());
		file.close ();
		return ParseString (s);
	}

	Value parse_string (std::istringstream& ss)
	{
		char c;
		ss.ignore (1);
		std::size_t start = ss.tellg ();

		for (;;)
		{
			ss.get (c);
			if (ss.eof () || c == '"')
			{
				break;
			}
		}
		if (ss.eof ())
		{
			return Value ();
		}

		std::size_t length = static_cast<std::size_t> (ss.tellg ()) - start - 1;
		ss.seekg (start);

		std::string s;
		s.resize (length);
		ss.read (const_cast<char*> (s.data ()), length);
		ss.ignore (1);

		return Value (std::move (s));
	}

	Value parse_number (std::istringstream& ss)
	{
		std::int64_t i;
		size_t sspos = ss.tellg ();
		char c = ss.peek ();
		if (!ss.eof () && c != '.')
		{
			if (!(ss >> i))
			{
				return Value ();
			}
		}
		c = ss.peek ();
		if (!ss.eof () && (c == '.' || c == 'e' || c == 'E'))
		{
			ss.seekg (sspos);
			double d;
			if (!(ss >> d))
			{
				return Value ();
			}
			return Value (d);
		}

		if (ss.eof ())
		{
			return Value ();
		}

		return Value (i);
	}

	void parse_ignore (std::istringstream& ss)
	{
		for (;;)
		{
			char c = ss.peek ();
			if (!ss.eof () && (c == ' ' || c == '\n' || c == '\t'))
			{
				ss.ignore (1);
			}
			else
			{
				return;
			}
		}
	}

	Value parse_array (std::istringstream& ss)
	{
		ss.ignore (1);
		parse_ignore (ss);
		char c = ss.peek ();
		if (!ss.eof () && c == ']')
		{
			ss.ignore (1);
			return Value (Array ());
		}

		Array a;
		while (!ss.eof ())
		{
			parse_ignore (ss);
			a.PushBack (parse_value (ss));
			parse_ignore (ss);
			ss.get (c);
			if (!ss.eof () && c == ']')
			{
				return Value (a);
			}
			if (!ss.eof () && c != ',')
			{
				return Value (a);
			}
		}

		return Value ();
	}

	Value parse_object (std::istringstream& ss)
	{
		ss.ignore (1);
		parse_ignore (ss);
		char c = ss.peek ();
		if (!ss.eof () && c == '}')
		{
			ss.ignore (1);
			return Value (Object ());
		}

		Object o;
		while (!ss.eof ())
		{
			parse_ignore (ss);
			c = ss.peek ();
			if (ss.eof () || c != '"')
			{
				return Value (o);
			}
			Value name = parse_string (ss);
			if (name.Type () == NIL)
			{
				return Value (o);
			}
			parse_ignore (ss);
			c = ss.get ();
			if (ss.eof () || c != ':')
			{
				return Value (o);
			}
			parse_ignore (ss);
			o [name.AsString ()] = parse_value (ss);
			parse_ignore (ss);
			c = ss.get ();
			if (!ss.eof () && c == '}')
			{
				return Value (o);
			}
			if (ss.eof () || c != ',')
			{
				return Value (o);
			}
		}

		return Value ();
	}

	Value parse_value (std::istringstream& ss)
	{
		parse_ignore (ss);
		char c = ss.peek ();
		if (ss.eof ())
		{
			return Value ();
		}

		if (c == '"')
		{
			return parse_string (ss);
		}
		
		if (c == '{')
		{
			return parse_object (ss);
		}
		
		if (c == '[')
		{
			return parse_array (ss);
		}

		if (c == '#')
		{
			ss.ignore (1);
			c = ss.peek ();
			if (ss.eof () || c != '"')
			{
				return Value ();
			}
			Value fileName = parse_string (ss);
			if (fileName.Type () == NIL)
			{
				return Value ();
			}
			return ParseFile (fileName.AsString ());
		}

		if (c != 't' && c != 'f' && c != 'n')
		{
			return parse_number (ss);
		}

		if (c == 't')
		{
			std::string s;
			s.resize (4);
			ss.read (const_cast<char*> (s.data ()), 4);
			if (strcmp (s.c_str (), "true") == 0)
			{
				return Value (true);
			}
		}

		if (c == 'f')
		{
			std::string s;
			s.resize (5);
			ss.read (const_cast<char*> (s.data ()), 5);
			if (strcmp (s.c_str (), "false") == 0)
			{
				return Value (false);
			}
		}

		ss.ignore (4);
		return Value ();
	}

	Value ParseString (const std::string& s)
	{
		std::istringstream ss (s);

		return parse_value (ss);
	}

}
