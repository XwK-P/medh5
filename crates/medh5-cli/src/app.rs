//! The command grammar: every command, argument and help string.
//!
//! The grammar is the 1.x `argparse` parser's, argument for argument, so a
//! script written against 1.x runs unchanged: the same names, the same
//! defaults, the same `--flag` spellings (and, as `argparse` allowed, any
//! unambiguous prefix of a long flag).

use clap::{Arg, ArgAction, Command};

use medh5::annotations::encode::TRANSCODABLE;
use medh5::collection::SUFFIX;
use medh5::storage::index::{DEFAULT_MAX_COORDS, DEFAULT_OCCUPANCY_FACTOR};
use medh5::validate::LEVELS;

/// Codec profiles, sorted as `--profile` lists them.
pub const CODEC_PROFILES: [&str; 4] = ["archive", "balanced", "portable", "training"];
/// `scrub --profile` values.
pub const SCRUB_PROFILES: [&str; 2] = ["basic", "strict"];
/// How DICOM imports and 0.x migrations group files into samples.
pub const GROUPING: [&str; 2] = ["subject", "study"];
/// Fields worth grouping or stratifying on.
pub const GROUPABLE: [&str; 10] = [
    "subject_id",
    "group_id",
    "site_id",
    "scanner_id",
    "dataset_id",
    "acquisition_protocol",
    "sex",
    "laterality",
    "bodypart",
    "label_set_id",
];
/// `dataset split` ratios when none are given, as the help prints them.
pub const DEFAULT_RATIOS: &str = "train=0.7,val=0.15,test=0.15";

fn positional(name: &'static str, help: &'static str) -> Arg {
    Arg::new(name).help(help).required(true)
}

fn paths(help: &'static str) -> Arg {
    Arg::new("paths").value_name("PATH").num_args(1..).required(true).help(help)
}

fn flag(name: &'static str, long: &'static str, help: &'static str) -> Arg {
    Arg::new(name).long(long).action(ArgAction::SetTrue).help(help)
}

fn opt(name: &'static str, long: &'static str, help: &'static str) -> Arg {
    Arg::new(name).long(long).value_name(long.to_uppercase().replace('-', "_")).help(help)
}

fn append(name: &'static str, long: &'static str, help: &'static str) -> Arg {
    opt(name, long, help).action(ArgAction::Append)
}

fn int(name: &'static str, long: &'static str, help: &'static str) -> Arg {
    opt(name, long, help).value_parser(clap::value_parser!(i64)).allow_negative_numbers(true)
}

fn json() -> Arg {
    flag("json", "json", "machine-readable output on stdout")
}

fn key() -> Arg {
    opt("key", "key", "sample key inside a .medh5c collection; omit for a single sample")
}

fn out_required(help: impl Into<String>) -> Arg {
    Arg::new("out").short('o').long("out").value_name("OUT").required(true).help(help.into())
}

fn group(name: &'static str, help: &'static str) -> Command {
    Command::new(name).about(help).subcommand_value_name("COMMAND").subcommand_help_heading("Commands")
}

fn report_args(cmd: Command) -> Command {
    cmd.arg(Arg::new("report").long("report").value_name("FILE").help("write the report as JSON")).arg(json())
}

/// The whole `medh5` command.
pub fn command() -> Command {
    let about =
        format!("medh5 {} --- tools for the MEDH5 {} medical imaging container", medh5::VERSION, medh5::FORMAT_VERSION);
    Command::new("medh5")
        .about(about)
        .version(format!("{} (format {})", medh5::VERSION, medh5::FORMAT_VERSION))
        .disable_version_flag(true)
        .arg(
            Arg::new("version")
                .long("version")
                .action(ArgAction::Version)
                .help("show program's version number and exit"),
        )
        .subcommand_value_name("COMMAND")
        .subcommand_help_heading("Commands")
        .infer_long_args(true)
        .disable_help_subcommand(true)
        .max_term_width(100)
        .subcommands(inspect())
        .subcommands(seg())
        .subcommand(labels())
        .subcommands(curation())
        .subcommand(dataset())
        .subcommands(convert())
        .subcommands(perf())
        .subcommand(conformance())
        .subcommand(clinical())
        .subcommand(task())
        .subcommand(cache())
}

fn inspect() -> Vec<Command> {
    vec![
        Command::new("info")
            .about("summarise a sample, or a collection")
            .arg(positional("path", "the sample or collection to summarise"))
            .arg(key())
            .arg(json()),
        Command::new("tree")
            .about("annotated object listing with spec roles")
            .arg(positional("path", "the sample or collection to list"))
            .arg(key())
            .arg(json()),
        Command::new("validate")
            .about("check conformance (spec §15)")
            .arg(paths("one or more files"))
            .arg(
                opt("level", "level", "how much to check: structural, semantic (default), integrity or strict")
                    .value_parser(LEVELS)
                    .default_value("semantic"),
            )
            .arg(append("profiles", "profile", "override the declared profiles; repeatable").value_name("PROFILE"))
            .arg(flag("verbose", "verbose", "print every diagnostic, not only the summary").short('v'))
            .arg(json()),
        Command::new("verify")
            .about("check digests and content_id (spec §13)")
            .arg(paths("one or more files"))
            .arg(key())
            .arg(append("partial", "partial", "verify only these objects; repeatable").value_name("OBJ"))
            .arg(json()),
        Command::new("fix")
            .about("rebuild derived data; restamp digests")
            .arg(paths("one or more files"))
            .arg(flag("rebuild_index", "rebuild-index", "recompute stale sampling indices (§14.3)"))
            .arg(flag("rewrite_digests", "rewrite-digests", "restamp digests over the current bytes --- see --reason"))
            .arg(opt("reason", "reason", "why the digests are being rewritten; recorded in the file"))
            .arg(opt("performed_by", "by", "who is making the change; recorded in provenance"))
            .arg(json()),
        Command::new("timeline")
            .about("timepoints and what belongs to each")
            .arg(positional("path", "the sample whose visits to list"))
            .arg(key())
            .arg(json()),
        Command::new("track")
            .about("join instance ids across timepoints")
            .arg(positional("path", "the sample whose instances to join across visits"))
            .arg(opt("class_key", "class", "restrict to one class"))
            .arg(key())
            .arg(json()),
    ]
}

fn seg() -> Vec<Command> {
    let seg = group("seg", "voxel-annotation tools")
        .subcommand(
            Command::new("stats")
                .about("per-class counts, overlap graph and encoding cost model")
                .arg(positional("path", "the sample holding the annotation"))
                .arg(positional("annotation", "the voxel annotation to measure"))
                .arg(json()),
        )
        .subcommand(
            Command::new("convert")
                .about("re-encode losslessly (spec §7.6)")
                .arg(positional("path", "the sample holding the annotation"))
                .arg(positional("annotation", "the voxel annotation to re-encode"))
                .arg(
                    opt("to", "to", "target encoding: labelmap, layers, bitmask, instances or probmap")
                        .required(true)
                        .value_parser(TRANSCODABLE),
                )
                .arg(flag("dry_run", "dry-run", "report the size delta, write nothing"))
                .arg(flag(
                    "drop_identity",
                    "drop-identity",
                    "allow instances -> a dense encoding, which loses every instance_id; recorded in the provenance",
                ))
                .arg(json()),
        );
    let index = group("index", "derived sampling caches (spec §14.3)").subcommand(
        Command::new("build")
            .about("build or refresh sampling indices")
            .arg(paths("one or more files"))
            .arg(
                int("max_coords", "max-coords", "how many foreground coordinates to sample per class")
                    .default_value(DEFAULT_MAX_COORDS.to_string()),
            )
            .arg(
                int("occupancy", "occupancy", "block size, in voxels per axis, of the coarse occupancy map")
                    .default_value(DEFAULT_OCCUPANCY_FACTOR.to_string()),
            )
            .arg(int("seed", "seed", "seed for the coordinate sample, so the index is reproducible").default_value("0"))
            .arg(json()),
    );
    vec![seg, index]
}

fn labels() -> Command {
    group("labels", "inspect label sets")
        .subcommand(
            Command::new("show")
                .about("print a file's label set")
                .arg(positional("path", "the sample whose label set to print"))
                .arg(json()),
        )
        .subcommand(
            Command::new("check")
                .about("report vocabulary drift across files")
                .arg(paths("the samples to compare label sets across"))
                .arg(json()),
        )
        .subcommand(
            group("registry", "bundled vocabularies")
                .subcommand(Command::new("list").about("list bundled vocabularies").arg(json())),
        )
}

fn curation() -> Vec<Command> {
    vec![
        Command::new("pack")
            .about(format!("bundle sample files into one {SUFFIX}"))
            .arg(paths("sample files to pack"))
            .arg(out_required(format!("output {SUFFIX} file")))
            .arg(
                append("keys", "key", "sample key for each source, in order; defaults to the file stem")
                    .value_name("KEY"),
            )
            .arg(json()),
        Command::new("unpack")
            .about("extract samples from a collection")
            .arg(positional("path", "the .medh5c collection to extract from"))
            .arg(out_required("output directory"))
            .arg(append("keys", "key", "extract only these keys").value_name("KEY"))
            .arg(json()),
        Command::new("ls")
            .about("list the samples in a collection")
            .arg(positional("path", "the .medh5c collection to list"))
            .arg(json()),
        Command::new("prov")
            .about("who produced what, and how good it is (§11)")
            .arg(positional("path", "the sample whose provenance to print"))
            .arg(json()),
        Command::new("agree")
            .about("measure agreement between two annotations")
            .arg(positional("path", "the sample holding both annotations"))
            .arg(positional("a", "first annotation id").value_name("A"))
            .arg(positional("b", "second annotation id").value_name("B"))
            .arg(opt("metric", "metric", "voxel agreement metric: dice (default) or iou").value_parser(["dice", "iou"]))
            .arg(
                opt("threshold", "threshold", "IoU threshold when matching objects (default 0.5)")
                    .value_parser(clap::value_parser!(f64))
                    .allow_negative_numbers(true),
            )
            .arg(flag("record", "record", "print the `quality.agreement` record this measurement produces"))
            .arg(json()),
        Command::new("scrub")
            .about("find identifiers in the container and attest to it (§11.4)")
            .arg(paths("one or more files"))
            .arg(
                opt("profile", "profile", "how hard to look: basic (default) or strict")
                    .value_parser(SCRUB_PROFILES)
                    .default_value("basic"),
            )
            .arg(flag("apply_changes", "apply", "act on the actionable findings; without it, nothing is written"))
            .arg(int(
                "date_shift_days",
                "date-shift-days",
                "shift dates by N days instead of dropping them, preserving intervals",
            ))
            .arg(opt("salt", "salt", "salt the UID pseudonyms; keep it to reproduce the mapping").default_value(""))
            .arg(opt("performed_by", "by", "who performed the de-identification; recorded in the file"))
            .arg(flag(
                "pseudonymise_ids",
                "pseudonymise-ids",
                "with --apply and --salt: replace sample_id and subject_id with salted stable pseudonyms (they are \
                 often the record number)",
            ))
            .arg(json()),
        Command::new("splits")
            .about("audit split claims across files (§12.3)")
            .arg(paths("sample or collection files to audit"))
            .arg(json()),
    ]
}

fn dataset() -> Command {
    group("dataset", "cohort manifests, splits and statistics")
        .subcommand(
            Command::new("index")
                .about("metadata-only scan of a directory tree")
                .arg(positional("root", "directory tree to scan for .medh5 files"))
                .arg(out_required("manifest JSON to write"))
                .arg(flag("strict", "strict", "stop at the first unreadable file"))
                .arg(json()),
        )
        .subcommand(
            Command::new("split")
                .about("assign groups to partitions or folds")
                .arg(positional("manifest", "the manifest to split"))
                .arg(opt("out", "out", "split JSON to write").short('o'))
                .arg(
                    opt("set_id", "set-id", "name of this split, so several can coexist in one file")
                        .default_value("default"),
                )
                .arg(
                    Arg::new("group_by")
                        .long("group-by")
                        .value_name("GROUP_BY")
                        .default_value("group_id")
                        .help(format!("never a file; one of {}", GROUPABLE.join(", "))),
                )
                .arg(opt("stratify_by", "stratify-by", "entry field to balance across partitions"))
                .arg(int("k_folds", "k-folds", "produce k folds instead of ratio partitions"))
                .arg(
                    Arg::new("ratios")
                        .long("ratios")
                        .value_name("RATIOS")
                        .help(format!("e.g. train=0.8,val=0.2 (default {DEFAULT_RATIOS})")),
                )
                .arg(int("seed", "seed", "deterministic given the manifest, seed and parameters").default_value("0"))
                .arg(flag(
                    "write_claims",
                    "write-claims",
                    "stamp each sample with its partition and this manifest's digest",
                ))
                .arg(int("fold", "fold", "with --k-folds --write-claims: the validation fold"))
                .arg(opt("assigned_by", "assigned-by", "recorded on each claim as who assigned it"))
                .arg(json()),
        )
        .subcommand(
            Command::new("stats")
                .about("streaming intensity and class statistics")
                .arg(positional("manifest", "the manifest to compute over"))
                .arg(opt("out", "out", "statistics JSON to write").short('o'))
                .arg(append("images", "image", "image ids to include; repeatable").value_name("IMAGE"))
                .arg(
                    append("annotations", "annotation", "annotation ids to include; repeatable")
                        .value_name("ANNOTATION"),
                )
                .arg(int("workers", "workers", "processes to read with").default_value("1"))
                .arg(
                    int("stride", "stride", "read every Nth slab along the first axis (approximate, opt-in)")
                        .default_value("1"),
                )
                .arg(flag(
                    "stored",
                    "stored",
                    "measure the values the files store instead of physical ones; by default each image's rescale \
                     is applied first, as the loaders do",
                ))
                .arg(opt("partition", "partition", "restrict to one partition of --set-id"))
                .arg(opt("set_id", "set-id", "which split --partition refers to").default_value("default"))
                .arg(json()),
        )
        .subcommand(
            Command::new("check")
                .about("cross-file consistency (C1xx codes)")
                .arg(positional("manifest", "the manifest to check"))
                .arg(opt("set_id", "set-id", "which split's claims to cross-check"))
                .arg(flag("deep", "deep", "re-read each content_id instead of trusting size and mtime"))
                .arg(json()),
        )
}

fn convert() -> Vec<Command> {
    let pairs = |name: &'static str, long: &'static str, metavar: &'static str, help: &'static str| {
        append(name, long, help).value_name(metavar)
    };
    let convert = group("convert", "import from and export to other formats")
        .subcommand(report_args(
            Command::new("from-nifti")
                .about("NIfTI volumes -> one sample")
                .arg(positional("out", "the .medh5 file to write"))
                .arg(pairs("image", "image", "NAME=PATH", "an image channel; repeatable").required(true))
                .arg(pairs("mask", "mask", "NAME=PATH", "a mask; repeatable"))
                .arg(pairs("modality", "modality", "NAME=CODE", "NAME=CODE, the modality for an image; repeatable"))
                .arg(
                    opt("coord_system", "coord-system", "world coordinate system to store: LPS (default) or RAS")
                        .value_parser(["LPS", "RAS"])
                        .default_value("LPS"),
                )
                .arg(
                    opt(
                        "fourth_axis",
                        "fourth-axis",
                        "what a 4-D series' extra axis is: time (cine, DCE, 4-D CT) or channel (multi-b-value DWI, \
                         multi-echo). auto reads the file and reports a guess where it cannot tell",
                    )
                    .value_parser(["auto", "time", "channel"])
                    .default_value("auto"),
                )
                .arg(flag(
                    "assume_geometry",
                    "assume-geometry",
                    "import a NIfTI that declares no spatial mapping (sform_code = qform_code = 0) by taking the \
                     pixdim fallback. Off by default: that grid is assumed, not measured. Recorded as a guess.",
                ))
                .arg(opt("sample_id", "sample-id", "sample id to write; defaults to the output filename"))
                .arg(opt("subject_id", "subject-id", "subject id to write")),
        ))
        .subcommand(
            Command::new("to-nifti")
                .about("one image or class -> NIfTI")
                .arg(positional("path", "the sample to read"))
                .arg(positional("image", "the image id to export"))
                .arg(positional("out", "the .nii.gz file to write"))
                .arg(opt("annotation", "annotation", "export this annotation instead of the image"))
                .arg(opt("class_key", "class", "with --annotation, the single class to export"))
                .arg(flag("stored", "stored", "skip the rescale")),
        )
        .subcommand(report_args(
            Command::new("from-dicom")
                .about("a DICOM tree -> samples")
                .arg(positional("root", "directory tree of DICOM files"))
                .arg(positional("out", "output file, or a directory when several samples result"))
                .arg(
                    opt("group_by", "group-by", "one sample per subject (default) or per study")
                        .value_parser(GROUPING)
                        .default_value("subject"),
                )
                .arg(
                    append("modalities", "modality", "only import these modalities; repeatable").value_name("MODALITY"),
                )
                .arg(
                    append("series_uids", "series", "only import these SeriesInstanceUIDs; repeatable")
                        .value_name("SERIES"),
                ),
        ))
        .subcommand(report_args(
            Command::new("from-dicom-seg")
                .about("a DICOM SEG -> an annotation")
                .arg(positional("seg", "the DICOM SEG file to import"))
                .arg(positional("sample", "the sample to add the annotation to"))
                .arg(opt("ann_id", "id", "id for the new annotation").default_value("seg"))
                .arg(opt("grid", "grid", "grid to place the frames on; inferred when omitted"))
                .arg(opt(
                    "frame_salt",
                    "frame-salt",
                    "the salt `medh5 scrub` pseudonymised the sample's frames with: the SEG's frame is compared as the \
                     scrub wrote it",
                )),
        ))
        .subcommand(report_args(
            Command::new("to-dicom-seg")
                .about("an annotation -> a DICOM SEG")
                .arg(positional("path", "the sample to read"))
                .arg(positional("annotation", "the annotation to export"))
                .arg(positional("out", "the DICOM SEG file to write"))
                .arg(append("source", "source", "a source DICOM file").required(true)),
        ))
        .subcommand(report_args(
            Command::new("from-rtstruct")
                .about("an RTSTRUCT -> contours")
                .arg(positional("rtstruct", "the RTSTRUCT file to import"))
                .arg(positional("sample", "the sample to add the contours to"))
                .arg(opt("ann_id", "id", "id for the new annotation").default_value("contours"))
                .arg(opt("grid", "grid", "grid the contours are measured against"))
                .arg(flag(
                    "rasterize",
                    "rasterize",
                    "also derive a voxel annotation; the rule is recorded in provenance",
                )),
        ))
        .subcommand(report_args(
            Command::new("to-rtstruct")
                .about("contours -> an RTSTRUCT")
                .arg(positional("path", "the sample to read"))
                .arg(positional("annotation", "the contour annotation to export"))
                .arg(positional("out", "the RTSTRUCT file to write"))
                .arg(append("source", "source", "a source DICOM file; repeat once per slice").required(true)),
        ))
        .subcommand(report_args(
            Command::new("from-nnunet")
                .about("an nnU-Net v2 dataset -> samples")
                .arg(positional("root", "the nnU-Net v2 dataset directory"))
                .arg(positional("out", "directory to write the samples into"))
                .arg(append("case_ids", "case", "only import these case ids; repeatable").value_name("CASE")),
        ))
        .subcommand(report_args(
            Command::new("to-nnunet")
                .about("samples -> an nnU-Net v2 dataset")
                .arg(positional("out", "directory to write the dataset into"))
                .arg(Arg::new("paths").num_args(1..).required(true).help("the samples to export"))
                .arg(
                    opt("dataset_name", "dataset-name", "nnU-Net dataset name, e.g. Dataset001_Liver")
                        .default_value("Dataset001_medh5"),
                )
                .arg(opt("annotation", "annotation", "the annotation to export as labels").default_value("seg"))
                .arg(
                    append(
                        "classes",
                        "class",
                        "export only this class (id or key), which every case examined; repeatable",
                    )
                    .value_name("K"),
                ),
        ));
    let migrate = report_args(
        Command::new("migrate")
            .about("0.x files -> 1.0 samples (Appendix B)")
            .arg(Arg::new("paths").num_args(1..).required(true).help("the 0.x files to convert"))
            .arg(out_required("output directory"))
            .arg(
                opt("group_by", "group-by", "merge files sharing --subject-key into one sample per subject")
                    .value_parser(GROUPING)
                    .default_value("study"),
            )
            .arg(opt("subject_key", "subject-key", "dotted path to a subject key in 0.x extra, e.g. extra.patient_id"))
            .arg(opt("label_set", "label-set", "a reviewed label-set sidecar to reuse (see --write-labels)"))
            .arg(
                opt("write_labels", "write-labels", "mint the cohort's label set, write it for review, and stop")
                    .value_name("FILE"),
            ),
    );
    vec![convert, migrate]
}

fn perf() -> Vec<Command> {
    vec![
        Command::new("recompress")
            .about("re-encode bulk data under another codec profile (§14.2)")
            .arg(paths("one or more files"))
            .arg(
                opt("profile", "profile", "codec profile: training, balanced, archive or portable")
                    .required(true)
                    .value_parser(CODEC_PROFILES),
            )
            .arg(flag(
                "rechunk",
                "rechunk",
                "also re-derive chunk shapes; off by default, since a codec change is not an access-pattern change",
            ))
            .arg(opt("out", "out", "write beside the source instead of replacing it (single input only)").short('o'))
            .arg(json()),
        Command::new("bench")
            .about("reproduce the performance targets")
            .arg(Arg::new("path").help("a sample to measure; omitted, a synthetic one is written and used"))
            .arg(opt("annotation", "annotation", "annotation to time label reads against"))
            .arg(int("patch", "patch", "patch side length to measure with").default_value("64"))
            .arg(int("repeats", "repeats", "repetitions per measurement; the median is reported").default_value("20"))
            .arg(int("workers", "workers", "dataloader workers for the throughput run").default_value("0"))
            .arg(flag("no_throughput", "no-throughput", "skip the dataloader run (it needs PyTorch)"))
            .arg(json()),
    ]
}

fn conformance() -> Command {
    group("conformance", "the conformance corpus (spec §15)")
        .subcommand(Command::new("list").about("list corpus cases").arg(json()))
        .subcommand(
            Command::new("build")
                .about("write the corpus and its manifest")
                .arg(positional("outdir", "directory to write the corpus into"))
                .arg(append("names", "case", "only build these cases; repeatable").value_name("CASE")),
        )
        .subcommand(
            Command::new("run")
                .about("build the corpus and check this validator")
                .arg(positional("outdir", "directory to build the corpus in"))
                .arg(append("names", "case", "only run these cases; repeatable").value_name("CASE"))
                .arg(json()),
        )
        .subcommand(
            Command::new("publish")
                .about("write the distributable suite: cases, codes, schema, checksums")
                .arg(positional("outdir", "directory to write the distributable suite into"))
                .arg(append("names", "case", "only publish these cases; repeatable").value_name("CASE")),
        )
        .subcommand(
            Command::new("score")
                .about("score any implementation's results against a published suite")
                .arg(positional("suite", "a directory written by `conformance publish`"))
                .arg(positional("results", "JSON: [{file, errors, warnings}, ...]"))
                .arg(json()),
        )
}

fn float(name: &'static str, long: &'static str, help: &'static str) -> Arg {
    opt(name, long, help).value_parser(clap::value_parser!(f64)).allow_negative_numbers(true)
}

fn clinical() -> Command {
    let sample =
        |cmd: Command| cmd.arg(positional("path", "a MEDH5 1.1 sample (or collection, with --key)")).arg(key());
    group("clinical", "the clinical profile: events, documents and links on a subject clock (format 1.1)")
        .subcommand(
            sample(Command::new("show").about("the clock, events, documents and links of a sample")).arg(json()),
        )
        .subcommand(
            sample(Command::new("select").about("what strict prospective selection admits at a cutoff"))
                .arg(int("cutoff_us", "cutoff-us", "the cutoff, in microseconds on the subject clock"))
                .arg(float("cutoff_hours", "cutoff-hours", "the cutoff, in hours on the subject clock"))
                .arg(
                    opt("policy", "policy", "strict_prospective (default) or latest_provable")
                        .value_parser(medh5::clinical::POLICIES),
                )
                .arg(int("context_us", "context-us", "only events this long before the cutoff (closed boundary)"))
                .arg(opt("policy_file", "policy-file", "a JSON selection policy (task-and-cache contract §3.4)"))
                .arg(json()),
        )
        .subcommand(
            sample(Command::new("export").about("write the logical-record bundle (every text included) as JSON"))
                .arg(opt("out", "out", "write to this file instead of stdout").short('o')),
        )
        .subcommand(
            Command::new("augment")
                .about("add clinical records to a sample: in place, or into --out (1.1 §10)")
                .arg(positional("path", "the sample to augment"))
                .arg(positional("records", "a logical-record bundle (JSON)"))
                .arg(opt("out", "out", "write a new file instead of replacing the sample").short('o'))
                .arg(json()),
        )
        .subcommand(
            Command::new("strip")
                .about("write the imaging projection: the sample without its clinical profile (a reported loss)")
                .arg(positional("path", "the sample to project"))
                .arg(out_required("the new file (never the source)"))
                .arg(json()),
        )
}

fn task() -> Command {
    let manifest = || positional("manifest", "a medh5.task/1 manifest (JSON)");
    group("task", "task manifests: rows, cutoffs, sources and what they admit (medh5.task/1)")
        .subcommand(
            Command::new("validate")
                .about("check a manifest without opening its sources (T1xx, T2xx)")
                .arg(manifest())
                .arg(json()),
        )
        .subcommand(
            Command::new("preflight")
                .about("open and check every source, and report each row: eligible, uncertifiable, excluded or error")
                .arg(manifest())
                .arg(opt("base", "base", "resolve relative source URIs here (default: the manifest's directory)"))
                .arg(flag("deep", "deep", "re-verify every dataset of every source, not only the clinical ones"))
                .arg(json()),
        )
        .subcommand(
            Command::new("reconcile")
                .about("record the events several fragments of a subject share, and their digests")
                .arg(manifest())
                .arg(opt("base", "base", "resolve relative source URIs here (default: the manifest's directory)"))
                .arg(opt("out", "out", "write the reconciled manifest here (default: in place)").short('o')),
        )
}

fn cache() -> Command {
    group("cache", "feature caches derived from MEDH5 sources (medh5.cache/1)").subcommand(
        Command::new("validate")
            .about("check a cache's checksums and pins, and --- given a task --- its admissibility (T4xx)")
            .arg(positional("path", "a .medh5cache file"))
            .arg(opt("task", "task", "also check the cache against this task manifest (T404-T406)"))
            .arg(opt("base", "base", "resolve the entries' relative source URIs here (default: the cache's directory)"))
            .arg(json()),
    )
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_grammar_is_consistent() {
        super::command().debug_assert();
    }
}
