import os


data2path = {
    "peS2o": "/proj/long-multi/zfchen/data/pile/peS2o/data/v2",
    "patent": "/proj/long-multi/zfchen/data/pile/Patents/data/dataset=uspto",
    "freelaw": "/proj/long-multi/zfchen/data/pile/freelaw",
    "arxiv": "/proj/long-multi/zfchen/data/pile/arxiv/data/dataset=arxiv",
    "cosmopedia": "/proj/long-multi/zfchen/data/pile/cosmopedia/data",
    "financial_research_papers": "/proj/long-multi/zfchen/data/pile/financial_research_papers",
    "paper_with_code": "/proj/long-multi/zfchen/data/pile/paper_with_code",
    "cybersecurity": "/proj/long-multi/zfchen/data/pile/cybersecurity2/data/version=2.3.2/dataset=cybersecurity",
    "sap": "/proj/long-multi/zfchen/data/pile/sap/data/version=1.0",
    "ibm-redbooks": "/proj/long-multi/zfchen/data/pile/ibm_redbooks/data/version=2.2.2/dataset=ibm-redbooks",
    "ibm.com": "/proj/long-multi/zfchen/data/pile/ibm.com/data/version=1.0.1",
    "superknowa": "/proj/long-multi/zfchen/data/pile/superknowa/data",
    "open-web-math": "/proj/long-multi/zfchen/data/pile/open-web-math/data",
    "algebraic-stack": "/proj/long-multi/zfchen/data/pile/algebraic-stack/train",
    "stackexchange": "/proj/long-multi/zfchen/data/pile/stackexchange/data/dataset=stackexchange",
    "multilang-wikipedia": "/proj/long-multi/zfchen/data/pile/multilang-wikipedia",
    "multilang-webhose": "/proj/long-multi/zfchen/data/pile/multilang-webhose",
    "edgar": "/proj/long-multi/zfchen/data/pile/edgar/data/dataset=EDGAR",
}

def get_directory_size(directory):
    total_size = 0
    # Walk the directory tree
    for dirpath, dirnames, filenames in os.walk(directory):
        # Sum the size of each file
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # Skip if it's a symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def format_large_number(number):
    # Option 1: Comma-separated format
    comma_format = "{:,}".format(number)
    
    # Option 2: Compact format with suffixes
    if abs(number) >= 1_000_000_000:
        compact_format = f"{number / 1_000_000_000:.1f}B"
    elif abs(number) >= 1_000_000:
        compact_format = f"{number / 1_000_000:.1f}M"
    elif abs(number) >= 1_000:
        compact_format = f"{number / 1_000:.1f}K"
    else:
        compact_format = str(number)
    
    return comma_format, compact_format

data2size = {}
for data, path in data2path.items():
    size = get_directory_size(path)
    size_str = format_large_number(size)[0]
    data2size[data] = size_str
    
print(data2size)