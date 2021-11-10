import simfin as sf
#%% download income statements
# Set your SimFin+ API-key for downloading data.
sf.set_api_key('O0g6w0UlQ91ftTWoNHeDWLb5IKhbUEbF')

# Set the local directory where data-files are stored.
# The directory will be created if it does not already exist.
sf.set_data_dir('~/simfin_data/')

# Download the data from the SimFin server and load into a Pandas DataFrame.
a = sf.load_income(variant='annual', market='us')
#%% download balance sheet