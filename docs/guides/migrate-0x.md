# Migrate from 0.x

1.0 ships a *reader* for the 0.x layout, not an implementation of it. `medh5
migrate` is a one-way door: convert once, review the result, and keep the
originals until you have.

0.x files are not readable by 1.0 and are not meant to be. A 0.x reader opening
a 1.0 file fails on the missing `schema_version`, which is the correct loud
failure.

## 1. Mint a label set, and review it before converting anything

0.x had mask *names*, not classes. Converting each file independently would mint
ids per file, and `liver` would be id 1 in one sample and id 2 in the next — a
cohort that looks fine file by file and is incoherent as a whole.

So do it in two passes. The first writes a label set and converts nothing you
have to keep:

```bash
medh5 migrate old/*.medh5 -o new/ --write-labels labels.json
```

Open `labels.json`. Fix the keys, merge the synonyms (`liver`, `Liver`,
`liver_1`), assign the ids you want, and add the hierarchy if the names imply
one. **This is the step that is worth an hour**, because every later decision
references these ids.

## 2. Convert the cohort against it

```bash
medh5 migrate old/*.medh5 -o new/ --label-set labels.json \
      --group-by subject --subject-key extra.patient_id --report migration.json
```

Ids are now minted once for the whole cohort.

## 3. Read the report

Four things are not mechanical, and each is reported per file.

**Voxel encoding.** 0.x stored one boolean volume per mask name. 1.0 measures the
overlap graph and picks an encoding — which changes the size and nothing else.

**Box corners.** 0.x boxes were slice-like integers `[min, max)`; 1.0 boxes sit
at voxel edges. Every corner shifts by −0.5, which is a real change in the
numbers and is reported as one. `[[2, 6]]` becomes `[[1.5, 5.5]]` and still
slices `2:6`.

**Label set.** Which names became which keys and ids.

**Grouping.** A 0.x file is study-scoped and carries no subject key, so the
default is one sample per file with a single `tp0`. `--group-by subject` merges
files sharing the key you name, ordering by date where there is one and by mtime
otherwise — and says which it used.

Instance correspondence is **never** inferred across merged files. Asserting
that lesion 2 at baseline is lesion 2 at follow-up would fabricate exactly the
tracking ground truth §7.4 exists to record. If you know the correspondence,
write it yourself with `instance_id`.

## 4. Check what you got

```bash
medh5 validate new/*.medh5 --level strict
medh5 dataset index new/ -o cohort.json
medh5 dataset check cohort.json
```

`dataset check` is the one that catches the cohort-level mistakes step 1 exists
to prevent — `C101` if more than one label set survived, `C102` if a class id
means different things in different files.

If you merged by subject, confirm the timepoints look right:

```bash
medh5 timeline new/case_0001.medh5
```

## Related

- **[Converters](../reference/converters.md)** — every `convert` command.
- **[Cohort check codes](../reference/cohort-checks.md)** — `C101` and `C102`.
- **[Specification Appendix B](../spec/medh5-1.0.md)** — the 0.x → 1.0 mapping.
