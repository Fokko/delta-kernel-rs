use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, quote_spanned, ToTokens};
use syn::spanned::Spanned;
use syn::{
    parse_macro_input, Attribute, Data, DataStruct, DeriveInput, Error, Field, Fields, Item, Lit,
    Meta, PathArguments, Type, Visibility,
};

/// Parses a dot-delimited column name into an array of field names. See
/// `delta_kernel::expressions::column_name::column_name` macro for details.
#[proc_macro]
pub fn parse_column_name(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let is_valid = |c: char| c.is_ascii_alphanumeric() || c == '_' || c == '.';
    let err = match syn::parse(input) {
        Ok(syn::Lit::Str(name)) => match name.value().chars().find(|c| !is_valid(*c)) {
            Some(bad_char) => Error::new(name.span(), format!("Invalid character: {bad_char:?}")),
            _ => {
                let path = name.value();
                let path = path.split('.').map(proc_macro2::Literal::string);
                return quote_spanned! { name.span() => [#(#path),*] }.into();
            }
        },
        Ok(lit) => Error::new(lit.span(), "Expected a string literal"),
        Err(err) => err,
    };
    err.into_compile_error().into()
}

/// Derive a `delta_kernel::schemas::ToSchema` implementation for the annotated struct. The actual
/// field names in the schema (and therefore of the struct members) are all mandated by the Delta
/// spec, and so the user of this macro is responsible for ensuring that
/// e.g. `Metadata::schema_string` is the snake_case-ified version of `schemaString` from [Delta's
/// Change Metadata](https://github.com/delta-io/delta/blob/master/PROTOCOL.md#change-metadata)
/// action (this macro allows the use of standard rust snake_case, and will convert to the correct
/// delta schema camelCase version).
///
/// If a field sets `allow_null_container_values`, it means the underlying data can contain null in
/// the values of the container (i.e. a `key` -> `null` in a `HashMap`). Therefore the schema should
/// mark the value field as nullable, but those mappings will be dropped when converting to an
/// actual rust `HashMap`. Currently this can _only_ be set on `HashMap` fields.
#[proc_macro_derive(
    ToSchema,
    attributes(allow_null_container_values, field_id, skip_schema)
)]
pub fn derive_to_schema(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_ident = input.ident;

    let schema_fields = gen_schema_fields(&input.data);
    let output = quote! {
        #[automatically_derived]
        impl delta_kernel::schema::ToSchema for #struct_ident {
            fn to_schema() -> delta_kernel::schema::StructType {
                use delta_kernel::schema::derive_macro_utils::{
                    ToDataType as _, GetStructField as _, GetNullableContainerStructField as _,
                };
                delta_kernel::schema::StructType::new_unchecked([
                    #schema_fields
                ])
            }
        }
    };
    proc_macro::TokenStream::from(output)
}

// turn our struct name into the schema name, goes from snake_case to camelCase
fn get_schema_name(name: &Ident) -> Ident {
    let snake_name = name.to_string();
    let mut next_caps = false;
    let ret: String = snake_name
        .chars()
        .filter_map(|c| {
            if c == '_' {
                next_caps = true;
                None
            } else if next_caps {
                next_caps = false;
                // This assumes we're using ascii, should be okay
                Some(c.to_ascii_uppercase())
            } else {
                Some(c)
            }
        })
        .collect();
    Ident::new(&ret, name.span())
}

/// Helper function to create field_id related errors
fn field_id_error(span: Span, message: &str) -> Error {
    Error::new(span, format!("field_id error: {}", message))
}

fn get_field_id(field_attributes: &[Attribute]) -> Result<Option<i64>, Error> {
    field_attributes
        .iter()
        .filter_map(|attr| match &attr.meta {
            Meta::NameValue(nv) => Some(nv),
            _ => None,
        })
        .find(|nv| matches!(nv.path.get_ident(), Some(ident) if ident == "field_id"))
        .map(|nv| {
            let span = nv.value.span();
            match &nv.value {
                syn::Expr::Lit(syn::ExprLit {
                    lit: Lit::Int(lit_int),
                    ..
                }) => lit_int.base10_parse().map_err(|e| {
                    field_id_error(lit_int.span(), &format!("Failed to parse integer: {}", e))
                }),
                _ => Err(field_id_error(span, "Expected field-id to be an integer")),
            }
        })
        .transpose() // Convert Option<Result<T, E>> to Result<Option<T>, E>
}

/// Check if a path segment is `Option<HashMap<K, V>>`.
fn is_option_of_hashmap(seg: &syn::PathSegment) -> bool {
    if seg.ident != "Option" {
        return false;
    }
    let PathArguments::AngleBracketed(angle_args) = &seg.arguments else {
        return false;
    };
    // Option has exactly one type argument
    let Some(syn::GenericArgument::Type(Type::Path(inner_type))) = angle_args.args.first() else {
        return false;
    };
    // Check if the inner type's last segment is HashMap
    inner_type
        .path
        .segments
        .last()
        .is_some_and(|seg| seg.ident == "HashMap")
}

fn gen_schema_field(field: &Field) -> TokenStream {
    let name = get_schema_name(field.ident.as_ref().unwrap());
    let have_schema_null = field.attrs.iter().any(|attr| {
        // check if we have allow_null_container_values attr
        match &attr.meta {
            Meta::Path(path) => path
                .get_ident()
                .is_some_and(|ident| ident == "allow_null_container_values"),
            _ => false,
        }
    });

    match field.ty {
        Type::Path(ref type_path) => {
            // Convert the type path segments into a single quoted string
            let type_path_quoted = type_path.path.segments.iter().map(|segment| {
                let segment_ident = &segment.ident;
                match &segment.arguments {
                    PathArguments::None => quote! { #segment_ident :: },
                    PathArguments::AngleBracketed(angle_args) => {
                        quote! { #segment_ident::#angle_args :: }
                    }
                    _ => Error::new(
                        segment.arguments.span(),
                        "Can only handle <> type path args",
                    )
                    .to_compile_error(),
                }
            });

            // First, determine which base function to call based on schema_null setting
            let base_call = if have_schema_null {
                if let Some(last_seg) = type_path.path.segments.last() {
                    let is_valid = last_seg.ident == "HashMap" || is_option_of_hashmap(last_seg);
                    if !is_valid {
                        return Error::new(
                            last_seg.ident.span(),
                            format!(
                                "Can only use allow_null_container_values on HashMap or \
                                 Option<HashMap> fields, not {}",
                                last_seg.ident
                            ),
                        )
                        .to_compile_error();
                    }
                }
                quote_spanned! { field.span() => #(#type_path_quoted)* get_nullable_container_struct_field(stringify!(#name)) }
            } else {
                quote_spanned! { field.span() => #(#type_path_quoted)* get_struct_field(stringify!(#name)) }
            };

            // Then, add field-id metadata if present
            match get_field_id(&field.attrs) {
                Ok(Some(id)) => {
                    quote_spanned! { field.span() => #base_call.add_metadata([(delta_kernel::schema::ColumnMetadataKey::ParquetFieldId.as_ref(), #id)]) }
                }
                Ok(None) => quote_spanned! { field.span() => #base_call },
                Err(err) => err.to_compile_error(),
            }
        }
        _ => Error::new(field.span(), format!("Can't handle type: {:?}", field.ty))
            .to_compile_error(),
    }
}

fn has_skip_schema(field: &Field) -> bool {
    field.attrs.iter().any(|attr| match &attr.meta {
        Meta::Path(path) => path.get_ident().is_some_and(|ident| ident == "skip_schema"),
        _ => false,
    })
}

fn gen_schema_fields(data: &Data) -> TokenStream {
    let fields = match data {
        Data::Struct(DataStruct {
            fields: Fields::Named(fields),
            ..
        }) => &fields.named,
        _ => {
            return Error::new(
                Span::call_site(),
                "this derive macro only works on structs with named fields",
            )
            .to_compile_error()
        }
    };

    let schema_fields = fields
        .iter()
        .filter(|f| !has_skip_schema(f))
        .map(gen_schema_field);

    quote! { #(#schema_fields),* }
}

/// Derive an IntoEngineData trait for a struct that has all fields implement `TryInto<Scalar>`.
///
/// This is a relatively simple macro to produce the boilerplate for converting a struct into
/// EngineData using the `create_one` method. TODO: (doc)tests included in the delta_kernel crate:
/// `IntoEngineData` trait.
#[proc_macro_derive(IntoEngineData)]
pub fn into_engine_data_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let struct_name = &input.ident;

    let Data::Struct(DataStruct {
        fields: Fields::Named(fields),
        ..
    }) = &input.data
    else {
        return Error::new(
            struct_name.span(),
            "IntoEngineData can only be derived for structs with named fields",
        )
        .to_compile_error()
        .into();
    };

    let fields = &fields.named;
    let field_idents = fields.iter().map(|f| &f.ident);
    let field_types: Vec<_> = fields.iter().map(|f| &f.ty).collect();

    let expanded = quote! {
        #[automatically_derived]
        impl delta_kernel::IntoEngineData for #struct_name
        where
            #(#field_types: TryInto<delta_kernel::expressions::Scalar>,)*
            #(delta_kernel::Error: From<<#field_types as TryInto<delta_kernel::expressions::Scalar>>::Error>,)*
        {
            fn into_engine_data(
                self,
                schema: delta_kernel::schema::SchemaRef,
                engine: &dyn delta_kernel::Engine)
            -> delta_kernel::DeltaResult<Box<dyn delta_kernel::EngineData>> {
                // NB: we `use` here to avoid polluting the caller's namespace
                use delta_kernel::EvaluationHandlerExtension as _;
                let values = [
                    #(self.#field_idents.try_into()?),*
                ];
                let evaluator = engine.evaluation_handler();
                evaluator.create_one(schema, &values)
            }
        }
    };

    proc_macro::TokenStream::from(expanded)
}

/// Mark items as `internal_api` to make them public iff the `internal-api` feature is enabled.
///
/// NOTE: This macro does not support `mod` declarations because of nuances in how the mod expander
/// and proc macro system interact for non-inline modules such as `mod foo;`. Use explicit
/// cfg-gated `pub mod` / `pub(crate) mod` for module visibility control instead.
#[proc_macro_attribute]
pub fn internal_api(
    _attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(item as Item);

    // Create a version with public visibility for the unstable feature
    let public_version = make_public(input.clone());

    // The original item stays as-is for the non-unstable case
    let output = quote! {
        #[cfg(feature = "internal-api")]
        #public_version

        #[cfg(not(feature = "internal-api"))]
        #input
    };

    output.into()
}

fn make_public(mut item: Item) -> Item {
    /// Transforms the passed visibility to be `pub`. We pass the original span that the visibility
    /// came from, and attach it to the newly created pub token. This means that the compiler treats
    /// it as user-written code and normal lints apply. We want this because it allows us to catch
    /// "private_in_public" violations that are tricky to notice when just slapping
    /// `#[internal_api]` on something.
    fn set_pub(vis: &mut Visibility, span: Span) -> Result<(), syn::Error> {
        if matches!(vis, Visibility::Public(_)) {
            return Err(Error::new(
                vis.span(),
                "ineligible for #[internal_api]: item is already public",
            ));
        }
        *vis = Visibility::Public(syn::token::Pub { span });
        Ok(())
    }

    macro_rules! set_vis {
        ($item:ident) => {{
            let vis_span = $item.vis.span();
            set_pub(&mut $item.vis, vis_span)
        }};
    }

    let result = match &mut item {
        Item::Fn(f) => set_vis!(f),
        Item::Struct(s) => set_vis!(s),
        Item::Enum(e) => set_vis!(e),
        Item::Trait(t) => set_vis!(t),
        Item::Type(t) => set_vis!(t),
        Item::Use(u) => set_vis!(u),
        Item::Static(s) => set_vis!(s),
        Item::Const(c) => set_vis!(c),
        Item::Union(u) => set_vis!(u),
        // foreign mod, impl block, and all others not handled
        _ => Err(Error::new(
            item.span(),
            format!("unsupported item type for #[internal_api]: {item:?}"),
        )),
    };

    if let Err(err) = result {
        let error = err.to_compile_error();
        let mut tokens = item.to_token_stream();
        tokens.extend(error);
        return syn::parse_quote!(#tokens);
    }

    item
}

#[cfg(test)]
mod tests {
    use super::*;
    // Tests for field_id parsing and validation

    // Helper function to parse a struct and extract the field_id logic
    fn test_field_id_parsing(input: &str) -> Result<TokenStream, String> {
        let input = syn::parse_str::<DeriveInput>(input).map_err(|e| e.to_string())?;
        let tokens = gen_schema_fields(&input.data);
        Ok(tokens)
    }

    #[test]
    fn test_valid_field_id_parsing() {
        let input = r#"
            struct TestStruct {
                #[field_id = 123]
                valid_field: String,

                #[field_id = 456]
                another_valid_field: i32,

                normal_field: bool,
            }
        "#;

        let result = test_field_id_parsing(input);
        assert!(result.is_ok(), "Valid field_id should parse successfully");

        let tokens = result.unwrap();
        let token_string = tokens.to_string();

        // Should contain metadata for valid field_ids
        assert!(token_string.contains("123"));
        assert!(token_string.contains("456"));
    }

    #[test]
    fn test_field_id_edge_cases() {
        // Test with zero
        let input_zero = r#"
            struct TestStruct {
                #[field_id = 0]
                zero_field: String,
            }
        "#;
        let result = test_field_id_parsing(input_zero);
        assert!(result.is_ok(), "field_id = 0 should be valid");
        let token_stream = result.unwrap().to_string();
        assert!(
            token_stream.contains(
                "(delta_kernel :: schema :: ColumnMetadataKey :: ParquetFieldId . as_ref () , 0i64)"
            ),
            "Expected 0, found: {}",
            token_stream
        );

        // Test with negative number
        let input_negative = r#"
            struct TestStruct {
                #[field_id = -1]
                negative_field: String,
            }
        "#;
        let result = test_field_id_parsing(input_negative);
        assert!(result.is_ok(), "field_id = -1 should be valid");
        let token_stream = result.unwrap().to_string();
        assert!(
            token_stream.contains(
                "(delta_kernel :: schema :: ColumnMetadataKey :: ParquetFieldId . as_ref () , - 1i64)"
            ),
            "Expected -1, found: {}",
            token_stream
        );

        // Test with large number
        let input_large = r#"
            struct TestStruct {
                #[field_id = 9223372036854775807]
                large_field: String,
            }
        "#;
        let result = test_field_id_parsing(input_large);
        assert!(result.is_ok(), "Large field_id should be valid");
        let token_stream = result.unwrap().to_string();
        assert!(
            token_stream.contains(
                "(delta_kernel :: schema :: ColumnMetadataKey :: ParquetFieldId . as_ref () , 9223372036854775807i64)"
            ),
            "Expected 9223372036854775807, found: {}",
            token_stream
        );
    }
}
