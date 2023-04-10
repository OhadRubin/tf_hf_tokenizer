extern crate libc;
use libc::{c_char, c_int};
use std::ffi::CStr;
use std::str;
use tokenizers::tokenizer::{Tokenizer, EncodeInput};
use tokenizers::InputSequence;

fn process_strings(strings: Vec<String>) -> Result<Vec<Vec<i32>>, String> {
    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None).map_err(|e| e.to_string())?;
    let inputs: Vec<EncodeInput> = strings
        .iter()
        .map(|s| InputSequence::from(s.to_owned()).into())
        .collect();
    let encodings = tokenizer.encode_batch(inputs, false).unwrap();
    Ok(encodings
        .into_iter()
        .map(|e| e.get_ids().iter().map(|&id| id as i32).collect())
        .collect())
}

#[no_mangle]
pub extern "C" fn process_strings_wrapper(input: *const *const c_char, len: usize, output: *mut *mut c_int, output_len: *mut usize) -> c_int {
    let string_ptrs = unsafe { std::slice::from_raw_parts(input, len) };
    let string_vec: Vec<String> = string_ptrs
        .iter()
        .map(|&ptr| unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
        .collect();

    let result = match process_strings(string_vec) {
        Ok(res) => res,
        Err(_) => return -1,
    };

    let flat_result: Vec<i32> = result.into_iter().flatten().collect();
    unsafe {
        *output_len = flat_result.len();
        let output_array = libc::malloc(flat_result.len() * std::mem::size_of::<i32>()) as *mut c_int;
        output_array.copy_from_nonoverlapping(flat_result.as_ptr(), flat_result.len());
        *output = output_array;
    }
    0
}

#[cfg(test)]
use std::path::Path;

// Add the following test module at the end of the src/lib.rs file
#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to load test data
    fn load_test_data() -> Vec<String> {
        vec![
            "Hello, how are you?".to_string(),
            "This is a test.".to_string(),
            "Custom operation in TensorFlow.".to_string(),
        ]
    }

    #[test]
    fn test_process_strings() {
        let input_strings = load_test_data();

        let result = process_strings(input_strings).unwrap();
        assert_eq!(result.len(), 3); // Check if the output has the same length as the input

        // Check if the output values are as expected (you need to replace the expected values with the ones you expect from your tokenizer)
        assert_eq!(result[0], vec![101, 8667, 117, 1293, 1132, 1128, 136, 102]);
        assert_eq!(result[1], vec![101, 1188, 1110, 170, 2774, 119, 102]);
        assert_eq!(result[2], vec![101, 1904, 1759, 1107, 16778, 1884, 119, 102]);
    }
}