use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_uint};
use tokenizers::tokenizer::{Tokenizer, EncodeInput};
use tokenizers::InputSequence;

#[no_mangle]
pub extern "C" fn process_strings(
    strings: *const *const c_char,
    len: usize,
    result: *mut c_uint,
    result_len: usize,
) {
    let string_ptrs = unsafe { std::slice::from_raw_parts(strings, len) };
    let string_vec: Vec<String> = string_ptrs
        .iter()
        .map(|&ptr| unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
        .collect();

    // Load the tokenizer
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).unwrap();

// Process the vector of strings
    let inputs: Vec<EncodeInput> = string_vec.iter().map(|s| InputSequence::from(s.to_owned()).into()).collect();
    let encodings = tokenizer.encode_batch(inputs, false).unwrap();

    let mut offset = 0;
    for encoding in encodings {
        let ids = encoding.get_ids();
        if offset + ids.len() <= result_len {
            unsafe {
                std::ptr::copy_nonoverlapping(ids.as_ptr(), result.offset(offset as isize), ids.len());
            }
            offset += ids.len();
        } else {
            break;
        }
    }
}