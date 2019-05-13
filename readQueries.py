def readQueries():
    
    with open('Queries/queries.txt') as f:
  
        queries = f.read().splitlines()

    f.close()
    
    return queries