#include "json_st.h"

#include <cmath>

namespace Json
{
	Value::Value () : type_ (NIL) {}

	Value::Value (const std::int64_t i) : int_ (i), type_ (INT) {}

	Value::Value (const std::int32_t i) : int_ (static_cast<std::int64_t> (i)), type_ (INT) {}

	Value::Value (const std::uint64_t i) : int_ (static_cast<std::int64_t> (i)), type_ (INT) {}

	Value::Value (const std::uint32_t i) : int_ (static_cast<std::int64_t> (i)), type_ (INT) {}

	Value::Value (const double f) : float_ (f), type_ (FLOAT) {}

	Value::Value (const float f) : float_ (static_cast<double> (f)), type_ (FLOAT) {}

	Value::Value (const bool b) : bool_ (b), type_ (BOOL) {}

	Value::Value (const char* s) : string_ (s), type_ (STRING) {}

	Value::Value (const std::string& s) : string_ (s), type_ (STRING) {}

	Value::Value (const Object& o) : object_ (o), type_ (OBJECT) {}

	Value::Value (const Array& o) : array_ (o), type_ (ARRAY) {}

	Value::Value (std::string&& s) : string_ (std::move (s)), type_ (STRING) {}

	Value::Value (Object&& o) : object_ (std::move (o)), type_ (OBJECT) {}

	Value::Value (Array&& o) : array_ (std::move (o)), type_ (ARRAY) {}

	Value::Value (const Value& v)
	{
		switch (v.Type ())
		{
			/** Base types */
		case INT:
			int_ = v.int_;
			type_ = INT;
			break;

		case FLOAT:
			float_ = v.float_;
			type_ = FLOAT;
			break;

		case BOOL:
			bool_ = v.bool_;
			type_ = BOOL;
			break;

		case NIL:
			type_ = NIL;
			break;

		case STRING:
			string_ = v.string_;
			type_ = STRING;
			break;

			/** Compound types */
		case ARRAY:
			array_ = v.array_;
			type_ = ARRAY;
			break;

		case OBJECT:
			object_ = v.object_;
			type_ = OBJECT;
			break;

		}
	}

	Value::Value (Value&& v)
	{
		switch (v.Type ())
		{
			/** Base types */
		case INT:
			int_ = std::move (v.int_);
			type_ = INT;
			break;

		case FLOAT:
			float_ = std::move (v.float_);
			type_ = FLOAT;
			break;

		case BOOL:
			bool_ = std::move (v.bool_);
			type_ = BOOL;
			break;

		case NIL:
			type_ = NIL;
			break;

		case STRING:
			string_ = std::move (v.string_);
			type_ = STRING;
			break;

			/** Compound types */
		case ARRAY:
			array_ = std::move (v.array_);
			type_ = ARRAY;
			break;

		case OBJECT:
			object_ = std::move (v.object_);
			type_ = OBJECT;
			break;

		}
	}

	Value& Value::operator= (const Value& v)
	{
		switch (v.Type ())
		{
			/** Base types */
		case INT:
			int_ = v.int_;
			type_ = INT;
			break;

		case FLOAT:
			float_ = v.float_;
			type_ = FLOAT;
			break;

		case BOOL:
			bool_ = v.bool_;
			type_ = BOOL;
			break;

		case NIL:
			type_ = NIL;
			break;

		case STRING:
			string_ = v.string_;
			type_ = STRING;
			break;

			/** Compound types */
		case ARRAY:
			array_ = v.array_;
			type_ = ARRAY;
			break;

		case OBJECT:
			object_ = v.object_;
			type_ = OBJECT;
			break;

		}

		return *this;
	}

	Value& Value::operator= (Value&& v)
	{
		switch (v.Type ())
		{
			/** Base types */
		case INT:
			int_ = std::move (v.int_);
			type_ = INT;
			break;

		case FLOAT:
			float_ = std::move (v.float_);
			type_ = FLOAT;
			break;

		case BOOL:
			bool_ = std::move (v.bool_);
			type_ = BOOL;
			break;

		case NIL:
			type_ = NIL;
			break;

		case STRING:
			string_ = std::move (v.string_);
			type_ = STRING;
			break;

			/** Compound types */
		case ARRAY:
			array_ = std::move (v.array_);
			type_ = ARRAY;
			break;

		case OBJECT:
			object_ = std::move (v.object_);
			type_ = OBJECT;
			break;

		}

		return *this;
	}

	Value& Value::operator[] (const std::string& key)
	{
		return object_ [key];
	}

	const Value& Value::operator[] (const std::string& key) const
	{
		return object_ [key];
	}

	Value& Value::operator[] (std::size_t i)
	{
		return array_ [i];
	}

	const Value& Value::operator[] (std::size_t i) const
	{
		return array_ [i];
	}

	double Value::AsDouble () const
	{
		switch (type_)
		{
		case FLOAT:
			return float_;
		case INT: 
			return static_cast<double> (int_);
		case BOOL:
			return bool_ ? 1.0 : 0.0;
		case STRING:
		{
			return strtod (string_.c_str (), nullptr);
		}
		}

		return NAN;
	}
	double Value::AsDouble(double defaultValue) const
	{
		switch (type_)
		{
		case FLOAT:
			return float_;
		case INT:
			return static_cast<double> (int_);
		case BOOL:
			return bool_ ? 1.0 : 0.0;
		case STRING:
		{
			return strtod(string_.c_str(), nullptr);
		}
		}

		return defaultValue;
	}

	float Value::AsFloat () const
	{
		switch (type_)
		{
		case FLOAT:
			return static_cast<float> (float_);
		case INT: 
			return static_cast<float> (int_);
		case BOOL:
			return bool_ ? 1.0f : 0.0f;
		case STRING:
		{
			return strtof (string_.c_str (), nullptr);
		}
		}

		return NAN;
	}
	float Value::AsFloat(float defaultValue) const
	{
		switch (type_)
		{
		case FLOAT:
			return static_cast<float> (float_);
		case INT:
			return static_cast<float> (int_);
		case BOOL:
			return bool_ ? 1.0f : 0.0f;
		case STRING:
		{
			return strtof(string_.c_str(), nullptr);
		}
		}

		return defaultValue;
	}

	std::int64_t Value::AsInt64 () const
	{
		switch (type_)
		{
		case FLOAT:
			return static_cast<std::int64_t> (float_);
		case INT: 
			return static_cast<std::int64_t> (int_);
		case BOOL:
			return bool_ ? 1 : 0;
		case STRING:
		{
			return strtoll (string_.c_str (), nullptr, 10);
		}
		}

		return 0;
	}
	std::int64_t Value::AsInt64 (std::int64_t defaultValue) const
	{
		switch (type_)
		{
		case FLOAT:
			return static_cast<std::int64_t> (float_);
		case INT: 
			return static_cast<std::int64_t> (int_);
		case BOOL:
			return bool_ ? 1 : 0;
		case STRING:
		{
			return strtoll (string_.c_str (), nullptr, 10);
		}
		}

		return defaultValue;
	}

	std::int32_t Value::AsInt32 () const
	{
		switch (type_)
		{
		case FLOAT:
			return static_cast<std::int32_t> (float_);
		case INT: 
			return static_cast<std::int32_t> (int_);
		case BOOL:
			return bool_ ? 1 : 0;
		case STRING:
		{
			return strtol (string_.c_str (), nullptr, 10);
		}
		}

		return 0;
	}
	std::int32_t Value::AsInt32 (std::int32_t defaultValue) const
	{
		switch (type_)
		{
		case FLOAT:
			return static_cast<std::int32_t> (float_);
		case INT: 
			return static_cast<std::int32_t> (int_);
		case BOOL:
			return bool_ ? 1 : 0;
		case STRING:
		{
			return strtol (string_.c_str (), nullptr, 10);
		}
		}

		return defaultValue;
	}

	bool Value::AsBool () const
	{
		switch (type_)
		{
		case FLOAT:
			return float_ == 0.0 ? false : true;
		case INT: 
			return int_ == 0 ? false : true;
		case BOOL:
			return bool_;
		case STRING:
		{
			return string_.compare ("true") == 0 ? true : false;
		}
		}

		return false;
	}
	bool Value::AsBool (bool defaultValue) const
	{
		switch (type_)
		{
		case FLOAT:
			return float_ == 0.0 ? false : true;
		case INT: 
			return int_ == 0 ? false : true;
		case BOOL:
			return bool_;
		case STRING:
		{
			return string_.compare ("true") == 0 ? true : false;
		}
		}

		return defaultValue;
	}

	Object::Object () {}

	Object::~Object () {}

	Object::Object (const Object& o) : object_ (o.object_) {}

	Object::Object (Object&& o) : object_ (std::move (o.object_)) {}

	Object& Object::operator= (const Object& o)
	{
		object_ = o.object_;
		return *this;
	}

	Object& Object::operator= (Object&& o)
	{
		object_ = std::move (o.object_);
		return *this;
	}

	Value& Object::operator[] (const std::string& key)
	{
		return object_ [key];
	}

	const Value& Object::operator[] (const std::string& key) const
	{
		return object_.at (key);
	}

	std::pair<std::map<std::string, Value>::iterator, bool> Object::Insert (const std::pair<std::string, Value>& v)
	{
		return object_.insert (v);
	}

	std::map<std::string, Value>::const_iterator Object::Begin () const
	{
		return object_.begin ();
	}

	std::map<std::string, Value>::const_iterator Object::End () const
	{
		return object_.end ();
	}

	std::map<std::string, Value>::iterator Object::Begin ()
	{
		return object_.begin ();
	}

	std::map<std::string, Value>::iterator Object::End ()
	{
		return object_.end ();
	}

	std::size_t Object::Size () const
	{
		return object_.size ();
	}

	bool Object::Contains (const std::string& key) const
	{
		for (const auto& o : object_)
		{
			if (o.first.compare (key) == 0)
			{
				return true;
			}
		}

		return false;
	}

	Array::Array () {}

	Array::~Array () {}

	Array::Array (const Array& a) : array_ (a.array_) {}

	Array::Array (Array&& a) : array_ (std::move (a.array_)) {}

	Array& Array::operator= (const Array& a)
	{
		array_ = a.array_;
		return *this;
	}

	Array& Array::operator= (Array&& a)
	{
		array_ = std::move (a.array_);
		return *this;
	}

	Value& Array::operator[] (std::size_t i)
	{
		return array_.at (i);
	}

	const Value& Array::operator[] (size_t i) const
	{
		return array_.at (i);
	}

	std::vector<Value>::const_iterator Array::Begin () const
	{
		return array_.begin ();
	}

	std::vector<Value>::const_iterator Array::End () const
	{
		return array_.end ();
	}

	std::vector<Value>::iterator Array::Begin ()
	{
		return array_.begin ();
	}

	std::vector<Value>::iterator Array::End ()
	{
		return array_.end ();
	}

	size_t Array::Size () const
	{
		return array_.size ();
	}

	void Array::PushBack (const Value& v)
	{
		array_.push_back (v);
	}

	void Array::PushBack (const Value&& v)
	{
		array_.push_back (std::move (v));
	}

	void Indent (std::ostream& os)
	{
		for (std::int32_t i = 0; i < ind; i++)
		{
			os << "\t";
		}
	}

}

std::ostream& operator<< (std::ostream& os, const Json::Value& v)
{
	switch (v.Type ())
	{
	/** Base types */
	case Json::INT:
		os << (std::int64_t)v;
		break;

	case Json::FLOAT:
		os << (double)v;
		break;

	case Json::BOOL:
		os << ((bool)v ? "true" : "false");
		break;

	case Json::NIL:
		os << "null";
		break;

	case Json::STRING:
		os << '"' << (std::string)v << '"';
		break;

	/** Compound types */
	case Json::ARRAY:
		os << (Json::Array)v;
		break;

	case Json::OBJECT:
		os << (Json::Object)v;
		break;

	}
	return os;
}

std::ostream& operator<< (std::ostream& os, const Json::Object& o)
{
	os << "{" << std::endl;
	Json::ind++;

	for (auto e = o.Begin (), ee = o.End (); e != ee;)
	{
		Json::Indent (os);
		os << '"' << e->first << '"' << ": " << e->second;
		if (++e != o.End ())
		{
			os << ",";
		}
		os << std::endl;
	}

	Json::ind--;
	Json::Indent (os);
	os << "}";

	return os;
}

std::ostream& operator<< (std::ostream& os, const Json::Array& a)
{
	os << "[" << std::endl;
	Json::ind++;

	for (auto e = a.Begin (), ee = a.End (); e != ee;)
	{
		Json::Indent (os);
		os << (*e);
		if (++e != a.End ())
		{
			os << ",";
		}
		os << std::endl;
	}

	Json::ind--;
	Json::Indent (os);
	os << "]";

	return os;
}
