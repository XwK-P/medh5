//! A documents table whose `text` column holds offsets and bytes the input
//! chose, read as the validator and the selection read it: deferred
//! (`read_table_deferring`), then scanned in slabs (`TextColumn::scan`) and a
//! cell at a time.  Any refusal is fine; a crash or an unbounded read is a
//! finding.
#![no_main]

use std::path::PathBuf;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use medh5::clinical::columns::{read_table_deferring, Values, DOCUMENT_COLUMNS};
use medh5::clinical::TextColumn;
use medh5::hdf5;

#[derive(Arbitrary, Debug)]
struct Table {
    /// The `text` buffer.
    bytes: Vec<u8>,
    /// Offsets as stored, when there are no lengths.
    offsets: Vec<u64>,
    /// Each row's length: offsets that start at 0 and end at the buffer's
    /// length, so the reader gets past its structural checks.
    lengths: Option<Vec<u8>>,
    /// The buffer's chunk, in bytes, and whether it is compressed.
    chunk: u16,
    deflate: bool,
    /// Byte positions asked about, and the slab the scan reads through.
    probes: Vec<u64>,
    slab: u16,
}

fn scratch() -> PathBuf {
    let name = format!("medh5-fuzz-columns-{}.h5", std::process::id());
    let shm = PathBuf::from("/dev/shm");
    if shm.is_dir() {
        shm.join(name)
    } else {
        std::env::temp_dir().join(name)
    }
}

fn utf8_column(table: &hdf5::Group, name: &str, data: &[u8], offsets: &[u64], chunk: usize, deflate: bool) {
    let column = table.create_group(name).unwrap();
    let mut builder = column.new_dataset_builder().with_data(data);
    if !data.is_empty() {
        builder = builder.chunk(chunk.clamp(1, data.len()));
        if deflate {
            builder = builder.deflate(4);
        }
    }
    builder.create("data").unwrap();
    column.new_dataset_builder().with_data(offsets).create("offsets").unwrap();
}

fuzz_target!(|t: Table| {
    let path = scratch();
    let offsets: Vec<u64> = match &t.lengths {
        Some(lengths) => {
            let total = t.bytes.len() as u64;
            let mut at = 0u64;
            let mut offsets = vec![0];
            for n in lengths {
                at = (at + u64::from(*n)).min(total);
                offsets.push(at);
            }
            *offsets.last_mut().unwrap() = total;
            offsets
        }
        None => t.offsets.clone(),
    };
    let rows = offsets.len().saturating_sub(1);
    medh5::h5::init();
    {
        let file = hdf5::File::create(&path).unwrap();
        let table = file.create_group("clinical").unwrap().create_group("documents").unwrap();
        let ids: Vec<u8> = (0..rows).flat_map(|i| format!("d{i}").into_bytes()).collect();
        let mut id_offsets = vec![0u64];
        for i in 0..rows {
            id_offsets.push(id_offsets[i] + format!("d{i}").len() as u64);
        }
        utf8_column(&table, "document_id", &ids, &id_offsets, 64, false);
        let media: Vec<u8> = b"text/plain".repeat(rows);
        let media_offsets: Vec<u64> = (0..=rows as u64).map(|i| i * 10).collect();
        utf8_column(&table, "media_type", &media, &media_offsets, 64, false);
        utf8_column(&table, "text", &t.bytes, &offsets, usize::from(t.chunk), t.deflate);
    }
    let file = hdf5::File::open(&path).unwrap();
    let table = file.group("clinical/documents").unwrap();
    if let Ok(raw) = read_table_deferring(&table, "documents", &DOCUMENT_COLUMNS, false, &["text"]) {
        if let Some(Values::Deferred { offsets }) = raw.column("text").map(|c| c.values.clone()) {
            let column = TextColumn::open(&table, "text", offsets, "/clinical/documents/text").unwrap();
            let _ = column.scan(&t.probes, usize::from(t.slab));
            for i in 0..column.rows().min(64) {
                let _ = column.cell(i);
            }
        }
    }
    drop(table);
    drop(file);
    let _ = std::fs::remove_file(&path);
});
