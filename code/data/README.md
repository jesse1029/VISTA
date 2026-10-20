# Public input data

`public_usgs_inputs.zip` preserves the five input files used by the research
programs. `python prepare_inputs.py --out inputs` extracts them and checks every
file against `input_hashes.json`.

The two 2021 tables are the paired research inputs (`desis_2021.csv` and
`prisma_2021.csv`, 16,581 locations). The 2020 files retain the original table
headers; the program selects the documented paired acquisition by image ID and
location ID.

Original sources:

- Aneece et al. (2024), DESIS and PRISMA spectral library of agricultural crops
  for California's Central Valley in August 2021:
  [DOI 10.5066/P14CQACU](https://doi.org/10.5066/P14CQACU).
- Aneece and Thenkabail (2022), DESIS and PRISMA spectral library of agricultural
  crops in California's Central Valley in the 2020 Growing Season:
  [DOI 10.5066/P98LO5D4](https://doi.org/10.5066/P98LO5D4).

The USGS catalogs identify both releases as public-domain data:
[2021 catalog](https://data.usgs.gov/datacatalog/data/USGS%3A65d66ec7d34ec3e1801d280b)
and [2020 catalog](https://data.usgs.gov/datacatalog/data/USGS%3A62a91cc2d34ec53d2770f06d).
These data retain those terms; the repository's MIT license applies to the
VISTA code. Cite the original data releases when using the tables.
