from datasets import load_dataset

E2E_TRAIN = load_dataset('e2e_nlg', split='train')
E2E_VALID = load_dataset('e2e_nlg', split='validation')
E2E_TEST = load_dataset('e2e_nlg', split='test')

E2E_CLEANED_TRAIN = load_dataset('GEM/e2e_nlg', split='train')
E2E_CLEANED_VALID = load_dataset('GEM/e2e_nlg', split='validation')
E2E_CLEANED_TEST = load_dataset('GEM/e2e_nlg', split='test')