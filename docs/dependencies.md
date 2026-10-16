# Direct backend dependencies and notices

Project source is MIT. Dependencies retain their own licenses; this source archive does not relicense them. Cargo.lock records the complete resolved graph. The following direct backend license texts are included verbatim for reference.

| Package | License | Bundled text |
| --- | --- | --- |
| clarabel-0.11.1 | Apache-2.0 | [text](licenses/clarabel-0.11.1-LICENSE.txt) |
| nalgebra-0.33.2 | Apache-2.0 | [text](licenses/nalgebra-0.33.2-LICENSE.txt) |

PyO3 0.30.1 provides the Python binding (MIT OR Apache-2.0). Its transitive dependencies and all other build dependencies are resolved by Cargo.lock. Full upstream sources/licenses are obtained by Cargo during a source build; they are not vendored in this archive.

The Linux wheel validation uses a locally built artifact only. No wheels or native libraries are included in this source ZIP. Build prerequisites and backend selection are documented in README.md.
