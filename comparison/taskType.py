import mteb

benchmark = mteb.get_benchmark("MTEB(Europe, v1)")
tasks = benchmark.tasks

retrieval_tasks = [task for task in tasks if getattr(task.metadata, "type", None) == "MultilabelClassification"]
# retrieval_tasks = [task for task in tasks if task.__class__.__name__ == "AlloprofRetrieval"]


# Output file
output_path = "alloprof_metadata.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for task in retrieval_tasks:
        name = task.__class__.__name__
        dataset = task.metadata.dataset
        task_type = task.metadata.type
        domains = task.metadata.domains

        # Extract base language codes
        raw_langs = task.metadata.eval_langs
        base_langs = list({lang.split("-")[0] for lang in raw_langs})

        # Write formatted output
        f.write("=" * 100 + "\n")
        f.write(f"📌 Task: {name}\n")
        f.write(f"📚 Dataset: {dataset['path']}\n")
        f.write(f"🏷 Type: {task_type}\n")
        f.write(f"🌐 Languages: {base_langs}\n")
        f.write(f"🗂 Domains: {domains}\n")
        f.write(f"🔖 Language Type Declaration:\n{name}(name='{name}', languages={base_langs})\n")
        f.write("=" * 100 + "\n\n")

print(f"✅ Metadata written to {output_path}")