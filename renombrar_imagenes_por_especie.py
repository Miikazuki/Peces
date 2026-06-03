import csv
import os
import re
from pathlib import Path

BASE_DIR = Path('.')
CSV_PATH = BASE_DIR / 'especies_identificadas.csv'
IMG_DIR = BASE_DIR / 'imagenes_descargadas'
OUT_CSV_PATH = BASE_DIR / 'especies_identificadas_renombrado.csv'


def slugify_species(species: str) -> str:
    value = species.strip().lower()
    value = value.replace(' ', '_')
    value = re.sub(r'[^a-z0-9_]+', '', value)
    value = re.sub(r'_+', '_', value).strip('_')
    return value


rows_out = []
renamed = 0
skipped = 0
missing = 0

with CSV_PATH.open('r', encoding='utf-8-sig', newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        archivo = row['archivo']
        especie = row['especie']
        old_path = IMG_DIR / archivo

        # Solo renombrar filas que ya tienen especie identificada.
        if not especie or especie == 'Sin identificar':
            rows_out.append(row)
            skipped += 1
            continue

        if not old_path.exists():
            # Si el archivo no existe, dejamos la fila intacta.
            rows_out.append(row)
            missing += 1
            continue

        stem, ext = os.path.splitext(archivo)
        suffix_match = re.search(r'(\d+)$', stem)
        suffix = suffix_match.group(1) if suffix_match else '0000'

        species_slug = slugify_species(especie)
        new_name = f'{species_slug}_{suffix}{ext.lower()}'
        new_path = IMG_DIR / new_name

        # Evitar colisiones: si ya existe, agregar contador.
        if new_path.exists() and new_path.resolve() != old_path.resolve():
            count = 2
            while True:
                candidate = IMG_DIR / f'{species_slug}_{suffix}_{count}{ext.lower()}'
                if not candidate.exists():
                    new_path = candidate
                    new_name = candidate.name
                    break
                count += 1

        os.rename(old_path, new_path)

        row['archivo'] = new_name
        row['ruta_relativa'] = f'imagenes_descargadas/{new_name}'
        rows_out.append(row)
        renamed += 1

# Guardar CSV actualizado
with OUT_CSV_PATH.open('w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['archivo', 'ruta_relativa', 'link_relacionado', 'especie'])
    writer.writeheader()
    writer.writerows(rows_out)

print(f'Renombrados: {renamed}')
print(f'Sin identificar (no renombrados): {skipped}')
print(f'Faltantes en disco: {missing}')
print(f'CSV actualizado: {OUT_CSV_PATH.name}')
