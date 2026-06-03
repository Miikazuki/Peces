import csv
import re

# Leer output.csv usando csv.reader
url_to_species = {}

print("Leyendo output.csv...")
with open('output.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    headers = next(reader)  # Primera línea es header
    print(f"Headers encontrados: {len(headers)}")
    print(f"Headers: {headers}")
    
    # Buscar los índices de columnas importantes
    try:
        identifier_idx = headers.index('identifier')
        references_idx = headers.index('references')
        title_idx = headers.index('title')
        description_idx = headers.index('description')
        print(f"identifier_idx={identifier_idx}, references_idx={references_idx}, title_idx={title_idx}, description_idx={description_idx}")
    except ValueError as e:
        print(f"Error encontrando columnas: {e}")
        exit(1)
    
    # Procesar datos
    for line_num, row in enumerate(reader, start=2):
        if line_num <= 3 and len(row) >= max(identifier_idx, references_idx, title_idx) + 1:
            print(f"Row {line_num} tiene {len(row)} campos")
            if len(row) > identifier_idx:
                print(f"  identifier={row[identifier_idx][:50]if len(row[identifier_idx])>50 else row[identifier_idx]}")
            if len(row) > title_idx:
                print(f"  title={row[title_idx][:50] if len(row[title_idx])>50 else row[title_idx]}")
        
        if len(row) < max(identifier_idx, references_idx, title_idx) + 1:
            continue
        
        identifier = row[identifier_idx].strip()
        references = row[references_idx].strip()
        title = row[title_idx].strip()
        description = row[description_idx].strip() if len(row) > description_idx else ""
        
        # Extraer especie del title o description
        species = None
        for field_value in [title, description]:
            if field_value:
                match = re.search(r'([A-Z][a-z]+)\s+([a-z]+)', field_value)
                if match:
                    species = f"{match.group(1)} {match.group(2)}"
                    break
        
        if identifier and species:
            url_to_species[identifier] = species
        if references and species:
            url_to_species[references] = species

print(f"\nSe encontraron {len(url_to_species)} URLs con especies asignadas")
if len(url_to_species) > 0:
    print("Primeros ejemplos:")
    for url, sp in list(url_to_species.items())[:5]:
        short_url = (url[:50] + '...' if len(url) > 50 else url)
        print(f"  {short_url} -> {sp}")

# Leer no_especie_archivos.csv y crear salida con especies
results = []

print("Leyendo no_especie_archivos.csv y mapeando especies...")
with open('no_especie_archivos.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    headers_files = next(reader)
    archivo_idx = headers_files.index('archivo')
    link_idx = headers_files.index('link_relacionado')
    ruta_idx = headers_files.index('ruta_relativa')
    
    for row in reader:
        if len(row) < max(archivo_idx, link_idx, ruta_idx) + 1:
            continue
        
        archivo = row[archivo_idx].strip()
        link = row[link_idx].strip()
        ruta = row[ruta_idx].strip()
        
        # Buscar la especie basada en el link
        species_found = url_to_species.get(link, 'Sin identificar')
        
        results.append({
            'archivo': archivo,
            'ruta_relativa': ruta,
            'link_relacionado': link,
            'especie': species_found
        })

# Escribir resultados en nuevo CSV
print("Escribiendo resultados en especies_identificadas.csv...")
with open('especies_identificadas.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['archivo', 'ruta_relativa', 'link_relacionado', 'especie'])
    writer.writeheader()
    writer.writerows(results)

# Mostrar estadísticas
identificadas = sum(1 for r in results if r['especie'] != 'Sin identificar')
print(f"\nTotal de archivos: {len(results)}")
print(f"Especies identificadas: {identificadas}")
print(f"Sin identificar: {len(results) - identificadas}")
print(f"\nResultados guardados en: especies_identificadas.csv")

# Mostrar especies únicas encontradas
especies_unicas = set(r['especie'] for r in results if r['especie'] != 'Sin identificar')
print(f"\nEspecies encontradas ({len(especies_unicas)}):")
for sp in sorted(especies_unicas):
    count = sum(1 for r in results if r['especie'] == sp)
    print(f"  - {sp}: {count} imagen(es)")
