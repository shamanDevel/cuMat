#ifndef TOOLS_JSON_JSON_ST_HH
#define TOOLS_JSON_JSON_ST_HH

#include <cstdint>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <stack>

// based on: https://bitbucket.org/tunnuz/json/src/777310e626f1e3289d24089e7dd619bd200da6cf/json_st.hh?at=default
namespace Json
{
	/** Possible JSON type of a value (array, object, bool, ...). */
	enum ValueType
	{
		INT,        // JSON's int
		FLOAT,      // JSON's float 3.14 12e-10
		BOOL,       // JSON's boolean (true, false)
		STRING,     // JSON's string " ... " or (not really JSON) ' ... '
		OBJECT,     // JSON's object { ... }
		ARRAY,      // JSON's array [ ... ]
		NIL         // JSON's null
	};

	class Value;

	/** A JSON object, i.e., a container whose keys are strings, this
	is roughly equivalent to a Python dictionary, a PHP's associative
	array, a Perl or a C++ map (depending on the implementation). */
	class Object
	{
	public:

		/** Constructor. */
		Object ();

		/** Copy constructor.
			@param o object to copy from
			*/
		Object (const Object& o);

		/** Move constructor. */
		Object (Object&& o);

		/** Assignment operator.
			@param o object to copy from
			*/
		Object& operator= (const Object& o);

		/** Move operator.
			@param o object to copy from
			*/
		Object& operator= (Object&& o);

		/** Destructor. */
		~Object ();

		/** Subscript operator, access an element by key.
			@param key key of the object to access
			*/
		Value& operator[] (const std::string& key);

		/** Subscript operator, access an element by key.
			@param key key of the object to access
			*/
		const Value& operator[] (const std::string& key) const;

		/** Retrieves the starting iterator (const).
			@remark mainly for printing
			*/
		std::map<std::string, Value>::const_iterator Begin () const;

		/** Retrieves the ending iterator (const).
			@remark mainly for printing
			*/
		std::map<std::string, Value>::const_iterator End () const;

		/** Retrieves the starting iterator */
		std::map<std::string, Value>::iterator Begin ();

		/** Retrieves the ending iterator */
		std::map<std::string, Value>::iterator End ();

		/** Inserts a field in the object.
			@param v pair <key, value> to insert
			@return an iterator to the inserted object
			*/
		std::pair<std::map<std::string, Value>::iterator, bool> Insert (const std::pair<std::string, Value>& v);

		/** Size of the object. */
		std::size_t Size () const;

		bool Contains (const std::string& key) const;

	private:

		/** Inner container. */
		std::map<std::string, Value> object_;
	};

	/** A JSON array, i.e., an indexed container of elements. It contains
	JSON values, that can have any of the types in ValueType. */
	class Array
	{
	public:

		/** Default Constructor. */
		Array ();

		/** Destructor. */
		~Array ();

		/** Copy constructor.
			@param a the array to copy from
			*/
		Array (const Array& a);

		/** Assignment operator.
			@param a array to copy from
			*/
		Array& operator= (const Array& a);

		/** Move constructor.
			@param a the array to move from
			*/
		Array (Array&& a);

		/** Move assignment operator.
			@param a array to move from
			*/
		Array& operator= (Array&& a);

		/** Subscript operator, access an element by index.
			@param i index of the element to access
			*/
		Value& operator[] (size_t i);

		/** Subscript operator, access an element by index.
			@param i index of the element to access
			*/
		const Value& operator[] (size_t i) const;

		/** Retrieves the starting iterator (const).
			@remark mainly for printing
			*/
		std::vector<Value>::const_iterator Begin () const;

		/** Retrieves the ending iterator (const).
			@remark mainly for printing
			*/
		std::vector<Value>::const_iterator End () const;

		/** Retrieves the starting iterator. */
		std::vector<Value>::iterator Begin ();

		/** Retrieves the ending iterator */
		std::vector<Value>::iterator End ();

		/** Inserts an element in the array.
			@param n (a pointer to) the value to add
			*/
		void PushBack (const Value& n);
		void PushBack (const Value&& n);

		/** Size of the array. */
		std::size_t Size () const;

	protected:

		/** Inner container. */
		std::vector<Value> array_;

	};

	/** A JSON value. Can have either type in ValueTypes. */
	class Value
	{
	public:

		/** Default constructor (type = NIL). */
		Value ();

		/** Copy constructor. */
		Value (const Value& v);

		/** Constructor from int. */
		Value (const std::int64_t i);

		/** Constructor from int. */
		Value (const std::int32_t i);

		/** Constructor from int. */
		Value (const std::uint64_t i);

		/** Constructor from int. */
		Value (const std::uint32_t i);

		/** Constructor from float. */
		Value (const double f);

		/** Constructor from float. */
		Value (const float f);

		/** Constructor from bool. */
		Value (const bool b);

		/** Constructor from pointer to char (C-string).  */
		Value (const char* s);

		/** Constructor from STD string  */
		Value (const std::string& s);

		/** Constructor from pointer to Object. */
		Value (const Object& o);

		/** Constructor from pointer to Array. */
		Value (const Array& a);

		/** Move constructor. */
		Value (Value&& v);

		/** Move constructor from STD string  */
		Value (std::string&& s);

		/** Move constructor from pointer to Object. */
		Value (Object&& o);

		/** Move constructor from pointer to Array. */
		Value (Array&& a);

		/** Type query. */
		ValueType Type () const
		{
			return type_;
		}

		/** Subscript operator, access an element by key.
			@param key key of the object to access
			*/
		Value& operator[] (const std::string& key);

		/** Subscript operator, access an element by key.
			@param key key of the object to access
			*/
		const Value& operator[] (const std::string& key) const;

		/** Subscript operator, access an element by index.
			@param i index of the element to access
			*/
		Value& operator[] (std::size_t i);

		/** Subscript operator, access an element by index.
			@param i index of the element to access
			*/
		const Value& operator[] (std::size_t i) const;

		/** Assignment operator. */
		Value& operator=(const Value& v);

		/** Move operator. */
		Value& operator=(Value&& v);

		/** Cast operator for float */
		explicit operator double () const { return AsDouble (); }

		/** Cast operator for float */
		explicit operator float () const { return AsFloat (); }

		/** Cast operator for int */
		explicit operator std::int64_t () const { return AsInt64 (); }

		/** Cast operator for int */
		explicit operator std::int32_t () const { return AsInt32 (); }

		/** Cast operator for bool */
		explicit operator bool () const { return AsBool (); }

		/** Cast operator for string */
		explicit operator std::string () const { return string_; }

		/** Cast operator for Object */
		operator Object () const { return object_; }

		/** Cast operator for Object */
		operator Array () const { return array_; }

		/** Cast operator for double */
		double AsDouble () const;
		double AsDouble(double defaultValue) const;

		/** Cast operator for float */
		float AsFloat () const;
		float AsFloat(float defaultValue) const;

		/** Cast operator for int64_t */
		std::int64_t AsInt64 () const;
		std::int64_t AsInt64(std::int64_t defaultValue) const;

		/** Cast operator for int32_t */
		std::int32_t AsInt32 () const;
		std::int32_t AsInt32(std::int32_t defaultValue) const;

		/** Cast operator for bool */
		bool AsBool () const;
		bool AsBool(bool defaultValue) const;

		/** Cast operator for string */
		std::string AsString () const { return string_; }

		const Array& AsArray () const { return array_; }

		const Object& AsObject () const { return object_; }

	protected:

		double				float_;
		std::int64_t		int_;
		bool				bool_;
		std::string			string_;

		Object				object_;
		Array				array_;

		ValueType			type_;
	};

	/** Indentation counter */
	static std::int32_t ind;

	/** Print correct indentation before printing anything */
	static void Indent (std::ostream& os = std::cout);
}

/** Output operator for Values */
std::ostream& operator<< (std::ostream&, const Json::Value&);

/** Output operator for Objects */
std::ostream& operator<< (std::ostream&, const Json::Object&);

/** Output operator for Arrays */
std::ostream& operator<< (std::ostream&, const Json::Array&);

#endif