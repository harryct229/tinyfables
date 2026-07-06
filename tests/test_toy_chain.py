from pathlib import Path

from tinyfables.cli import main

FIXTURE = str(Path(__file__).parent / "fixtures" / "tiny_corpus.jsonl")


def test_toy_chain_tokenizer_prep_pretrain_generate(tmp_path, capsys):
    runs = tmp_path / "runs"
    tok_dir = runs / "tok"
    prep_dir = runs / "prep"
    ckpt = runs / "pretrain"

    (tmp_path / "tok.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\nvocab_size: 512\nseed: 0\ncompare_gpt2: false\n"
    )
    (tmp_path / "prep.yaml").write_text(
        f"source:\n  jsonl_path: {FIXTURE}\ntokenizer_dir: {tok_dir}\nwindow: 256\nseed: 0\n"
    )
    (tmp_path / "pre.yaml").write_text(
        f"prep_dir: {prep_dir}\ntokenizer_dir: {tok_dir}\n"
        "n_layer: 2\nn_head: 2\nd_model: 64\nn_ctx: 256\n"
        "batch_size: 4\nsteps: 100\nwarmup_steps: 10\nlr: 0.001\nseed: 0\ndevice: cpu\n"
    )

    assert main(["run", "tokenizer", "--config", str(tmp_path / "tok.yaml"), "--out", str(tok_dir)]) == 0
    assert main(["run", "prep", "--config", str(tmp_path / "prep.yaml"), "--out", str(prep_dir)]) == 0
    assert main(["run", "pretrain", "--config", str(tmp_path / "pre.yaml"), "--out", str(ckpt)]) == 0

    capsys.readouterr()  # clear the stage-completion prints
    rc = main([
        "generate",
        "--checkpoint", str(ckpt),
        "--tokenizer", str(tok_dir),
        "--character", "a shy octopus",
        "--setting", "a quiet tide pool",
        "--challenge", "doubting oneself",
        "--outcome", "a friend helps just in time",
        "--moral", "courage grows by small steps",
        "--max-new-tokens", "60",
        "--min-new-tokens", "8",
        "--seed", "0",
    ])
    assert rc == 0
    fable = capsys.readouterr().out.strip()
    assert len(fable) > 0
    assert (ckpt / "model.safetensors").exists()
    assert (ckpt / "manifest.json").exists()
