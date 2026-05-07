# Chemical Informatics Skill

## SMILES Processing
- Canonical SMILES for consistent representation
- Validate SMILES strings before processing
- Handle stereochemistry information appropriately
- Use RDKit for molecular manipulation

## Molecular Fingerprinting
- Morgan fingerprints for similarity analysis
- MACCS keys for substructure screening
- Tanimoto similarity for chemical comparison
- Fingerprint radius and bit length optimization

## Chemical Identifiers
- CAS number validation and normalization
- InChI/InChIKey for unique identification
- Handle multiple name formats (common, IUPAC, trade names)
- Cross-reference chemical databases

## RDKit Integration
```python
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# Safe molecule creation
mol = Chem.MolFromSmiles(smiles)
if mol is not None:
    canonical_smiles = Chem.MolToSmiles(mol)
    morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
```

## Data Quality
- Validate chemical structures before analysis
- Handle tautomers and stereoisomers consistently
- Check for impossible chemical structures
- Standardize molecular representations
