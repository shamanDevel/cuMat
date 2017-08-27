# Naming Convention

 - File names:
   File names should be CamelCase with the name matching the name of the main class (if possible)
   It should be guarded by #ifndef __CUMAT_FILE_NAME_H__
   
 - Class names:
   Class names are CamelCase (with a capital first letter)
   
 - Function / Method names:
   Function / method names are camelCase (with a lower first letter)
   There is no distinction between static and non-static methods
   
 - Class attribute:
   public class attributes: my_attribute
   protected / private attributes: my_attribute_
   
 - Template arguments:
   Either a single upper case character: T, U
   or if longer: _Row, _LongTemplateArgument
   
 - Function parameters:
   Parameters are named: my_parameter
   
 - Macros:
   Macros are in full upper case and always prefixed with CUMAT_ (since they have no namespace)
   CUMAT_ASSERT, CUMAT_NAMESPACE_BEGIN