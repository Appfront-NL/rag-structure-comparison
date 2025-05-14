import mteb
from collections import defaultdict

benchmark = mteb.get_benchmark("MTEB(Europe, v1)")
tasks = benchmark.tasks

type_to_tasks = defaultdict(list)
for task in tasks:
    task_type = task.metadata.type
    type_to_tasks[task_type].append(task)

# Save all output to this file
output_path = "tasks_info.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for task_type, task_list in type_to_tasks.items():
        f.write(f"\n=== Task Type: {task_type} ===\n")
        for task in task_list:
            name = getattr(task, "name", task.__class__.__name__)
            metadata = task.metadata
            languages = metadata.languages if hasattr(metadata, "languages") else []

            f.write(f"\n- Task: {name}\n")
            f.write(f"  Languages: {', '.join(languages)}\n")
            f.write(f"  Metadata:\n")
            f.write(f"    Dataset: {metadata.dataset}\n")
            f.write(f"    Description: {metadata.description}\n")
            f.write(f"    Type: {metadata.type}\n")
            f.write(f"    Modalities: {metadata.modalities}\n")
            f.write(f"    Category: {metadata.category}\n")
            f.write(f"    Reference: {metadata.reference}\n")
            f.write(f"    Eval Splits: {metadata.eval_splits}\n")

            # Handle eval_langs properly
            f.write(f"    Eval Langs:\n")
            if isinstance(metadata.eval_langs, dict):
                for lang_code, variants in metadata.eval_langs.items():
                    f.write(f"      {lang_code}: {variants}\n")
            elif isinstance(metadata.eval_langs, list):
                for lang in metadata.eval_langs:
                    f.write(f"      - {lang}\n")
            else:
                f.write(f"      (Unsupported type: {type(metadata.eval_langs).__name__})\n")

            f.write(f"    Main Score: {metadata.main_score}\n")
            f.write(f"    Date: {metadata.date}\n")
            f.write(f"    Domains: {metadata.domains}\n")
            f.write(f"    Task Subtypes: {metadata.task_subtypes}\n")
            f.write(f"    License: {metadata.license}\n")
            f.write(f"    Annotations Creators: {metadata.annotations_creators}\n")
            f.write(f"    Dialect: {metadata.dialect}\n")
            f.write(f"    Sample Creation: {metadata.sample_creation}\n")
            f.write(f"    Adapted From: {metadata.adapted_from}\n")
            f.write(f"    BibTeX Citation:\n{metadata.bibtex_citation}\n")

print(f"Task metadata written to: {output_path}")
