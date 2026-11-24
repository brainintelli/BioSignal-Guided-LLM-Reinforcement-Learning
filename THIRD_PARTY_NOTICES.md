# Third-party components and asset provenance

| Component | Source and rights |
|---|---|
| Qwen3.5-0.8B chat template | The unmodified [Jinja template](original/variants/273d8e0e683b/chat_template.jinja) is from the Qwen team’s [pinned upstream snapshot](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17), under [Apache License 2.0](docs/third-party/Apache-2.0.txt). The [per-file hash comparison](data/upstream/base-model.json) identifies the matching base weights, which are distributed upstream. |
| Godot, Python libraries and instrument software | The source references these separately installed runtimes. Follow each upstream distribution’s terms. The publicly included `maxlab.h` files are project simulation/test stubs. The commercial MaxLab SDK and instrument drivers require installation from their provider. |
| Game art and audio | The [asset manifest](original/project/assets/ASSET_MANIFEST.md) identifies the project-generated artwork, and the [audio manifest](original/project/assets/audio/manifest.json) records the original procedural compositions. |
| Scientific data and archived models | The public derived tables, raw recording entry and model assets have [separate terms](LICENSE-DATA). The project’s MIT grant applies to its own software and associated software documentation. |

Readers should retain the corresponding third-party notices when redistributing these components.
