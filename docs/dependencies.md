# Direct backend dependencies and notices

Project source is MIT. Dependencies keep their own licenses; this repository
does not relicense them. `Cargo.lock` records the complete resolved graph. The
license texts of the two direct numerical dependencies are included verbatim
for reference.

| Package | License | Text |
| --- | --- | --- |
| clarabel 0.11.1 | Apache-2.0 | [text](licenses/clarabel-0.11.1-LICENSE.txt) |
| nalgebra 0.33.2 | Apache-2.0 | [text](licenses/nalgebra-0.33.2-LICENSE.txt) |

Rayon 1.12.0 (MIT OR Apache-2.0) provides the bounded worker pools of the
`sqp-craft-parallel` crate; no global Rayon pool is configured.

PyO3 0.30.1 provides the Python binding (MIT OR Apache-2.0). Upstream sources
and licenses of all crates are fetched by Cargo during a source build; they are
not vendored. No wheels or native libraries are published (`publish = false`);
anyone distributing a built wheel must include the notices of the statically
linked crates.
