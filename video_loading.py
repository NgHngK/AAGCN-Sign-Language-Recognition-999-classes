import os
import csv
import re
import random
import collections
import copy
import pathlib

# Store the videos into videos_list.csv
class VideoCSVBuilder:
    def __init__(self, 
                 folder_path: str,
                 labels_file_path: str = '1_1000_label.csv',
                 csv_file_path: str = 'videos_list.csv',
                 final_file_path: str = 'temp_videos_list.csv'):
        """
        Initialize the builder with paths.
        """
        self.folder_path = folder_path
        self.labels_file_path = labels_file_path
        self.csv_file_path = csv_file_path
        self.final_file_path = final_file_path
        self.label_to_gloss = {}

    def create_video_list(self):
        """
        Build a mapping from labels to gloss and create
        the initial CSV of video files.
        """
        # --- build label->gloss dict ---
        with open(self.labels_file_path, mode='r', encoding='utf-8') as labels_file:
            csv_reader = csv.DictReader(labels_file)
            for row in csv_reader:
                try:
                    label = int(row['id_label_in_documents'])
                    gloss = row['name']
                    self.label_to_gloss[label] = gloss
                except (KeyError, ValueError, TypeError):
                    continue

        # --- write initial video list ---
        video_exts = ('.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv')
        with open(self.csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['file', 'label', 'gloss', 'video_name', 'actor'])

            for filename in os.listdir(self.folder_path):
                if filename.lower().endswith(video_exts):
                    actor = filename.split('_')[0] if '_' in filename else ''

                    match = re.search(r'_(\d+)\.', filename)
                    if match:
                        label = int(match.group(1))
                        gloss = self.label_to_gloss.get(label, 'Unknown')
                    else:
                        label = 'N/A'
                        gloss = 'Unknown'

                    full_filename = os.path.join(self.folder_path, filename)
                    csv_writer.writerow([full_filename, label, gloss, filename, actor])

        print(f'Video names have been written to {self.csv_file_path}')

    def normalize_labels(self):
        """
        Normalize labels so the minimum label becomes 0.
        """
        # --- find min label ---
        with open(self.csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            labels = []
            for row in csv_reader:
                lbl = row.get("label", "")
                if isinstance(lbl, str) and lbl.isdigit():
                    labels.append(int(lbl))
            min_label = min(labels) if labels else None

        print("Minimum label:", min_label)

        # --- normalize labels ---
        with open(self.csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file, \
             open(self.final_file_path, mode='w', newline='', encoding='utf-8') as final_file:

            csv_reader = csv.DictReader(csv_file)
            fieldnames = csv_reader.fieldnames or ['file', 'label', 'gloss', 'video_name', 'actor']
            csv_writer = csv.DictWriter(final_file, fieldnames=fieldnames)
            csv_writer.writeheader()

            for row in csv_reader:
                if min_label is not None and row.get('label', '').isdigit():
                    row['label'] = str(int(row['label']) - min_label)
                csv_writer.writerow(row)

        os.replace(self.final_file_path, self.csv_file_path)
        print("Labels have been updated and saved.")

    def store_video(self):
        """
        Run the full pipeline (create video list + normalize labels).
        """
        self.create_video_list()
        self.normalize_labels()

    def check_labels(self):
        """
        Print and return the number of unique labels in the current CSV.
        """
        labels = set()
        with open(self.csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                lbl = row.get("label", "")
                if lbl:
                    labels.add(lbl)

        print(labels)
        return len(labels)
    

'''
Some videos are missing in the dataset, making the number of videos of each label not the same.
For example, label 1 has the highest number of videos, which is 10.
We will duplicate videos of other labels (with fewer than 10) so that each label has 10 videos.
'''

def balance_videos_list(folder_path, input_csv="videos_list.csv", output_csv="videos_list_balanced.csv"):
    builder = VideoCSVBuilder(folder_path)
    builder.store_video()
    num_labels = builder.check_labels()

    def canonicalize_actor(s):
        s = str(s)
        m = re.match(r'^(\d{1,2})', s)
        return f"{int(m.group(1)):02d}" if m else s

    with open(input_csv, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        fieldnames = rows[0].keys()

    for r in rows:
        r['actor'] = canonicalize_actor(r['actor'])

    actors = sorted({r['actor'] for r in rows})
    total_actors = len(actors)
    print(f"Detected {total_actors} actors: {actors}")

    by_label = collections.defaultdict(list)
    for r in rows:
        by_label[r['label']].append(r)

    def patch_row(base_row, missing_actor):
        new_row = copy.deepcopy(base_row)
        new_actor_code = missing_actor
        new_row['actor'] = new_actor_code

        def swap_code(s):
            return re.sub(r'^\d{1,2}(?=[_-])', new_actor_code, s)

        new_row['video_name'] = swap_code(new_row['video_name'])
        new_row['file'] = swap_code(new_row['file'])
        return new_row

    for label, label_rows in list(by_label.items()):
        present_actors = {r['actor'] for r in label_rows} 
        missing = [a for a in actors if a not in present_actors]
        for ma in missing:
            donor = random.choice(label_rows)
            synthetic = patch_row(donor, ma)
            label_rows.append(synthetic)
        by_label[label] = label_rows

    final_rows = []
    for label, label_rows in by_label.items():
        seen, uniques, extras = {}, [], []
        for r in label_rows:
            actor = r['actor'] 
            if actor not in seen:
                seen[actor] = True
                uniques.append(r)
            else:
                extras.append(r)
        while len(uniques) < total_actors and extras:
            uniques.append(extras.pop())
        final_rows.extend(uniques[:total_actors])

    pathlib.Path(output_csv).write_text('', encoding='utf-8')
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"\nBalanced file written to: {output_csv}")
    counter = collections.Counter(r['label'] for r in final_rows)

    print("\nCount per label (ascending):")
    for lab, cnt in sorted(counter.items(), key=lambda x: x[1]):
        print(f"{lab:>4} : {cnt}")

    return num_labels
