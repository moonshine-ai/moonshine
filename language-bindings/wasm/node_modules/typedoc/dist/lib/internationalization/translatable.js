import translatable from "./locales/en.cjs";
// Compiler errors here which says a property is missing indicates that the value on translatable
// is not a literal string. It should be so that TypeDoc's placeholder replacement detection
// can validate that all placeholders have been specified.
({});
// Compiler errors here which says a property is missing indicates that the key on translatable
// contains a placeholder _0/_1, etc. but the value does not match the expected constraint.
translatable;
