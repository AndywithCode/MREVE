//Project: VulKG Project
//DBMA: Graph DBMS
//DATABASE: neo4j
//Password: Neo4j

// ###############  import/CVE_knowledge.csv  ####################
// create entity
// label: Vulnerability

// set uniqueness constraint
CREATE CONSTRAINT UniqueCveID ON (v:Vulnerability) ASSERT v.cveID IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

//create Entity with properties: Vulnerability (no relationships)
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///CVE_knowledge.csv')
 YIELD map AS row RETURN row",
 "WITH
 row.cveID AS cveID,
 row.publishedDate AS publishedDate,
 row.cveDescription AS cveDescription,
 row.commitMsg AS commitMsg
MERGE (v:Vulnerability {cveID:cveID})
    ON CREATE SET
    v.publishedDate=publishedDate,
    v.cveDescription=cveDescription,
    v.commitMsg=commitMsg
 RETURN count(*)",
 {batchSize: 500}
);

// ###############  import/CWE_knowledge.csv  ####################
// create entity
// label: Weakness

// set uniqueness constraint
CREATE CONSTRAINT UniquecweID ON (w:Weakness) ASSERT w.cweID IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

//create entity with properties: Weakness (no relationships)
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///CWE_knowledge.csv')
 YIELD map AS row RETURN row",
 "WITH
 row.cweID AS cweID,
 row.cweName	AS cweName,
 row.weaknessAbstraction AS weaknessAbstraction,
 row.status AS status,
 row.cweDescription AS cweDescription
 MERGE (w:Weakness {cweID:cweID})
    ON CREATE SET
	w.cweName = cweName,
	w.weaknessAbstraction = weaknessAbstraction,
	w.status = status,
	w.cweDescription = cweDescription
 RETURN count(*)",
 {batchSize: 500}
);

// create relationship
// relationship type: EXAMPLE_OF (Vulnerability EXAMPLE_OF Weakness)

CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///CVE_knowledge.csv')
 YIELD map AS row RETURN row",
 "WITH
	 row.cveID AS cveID,
	 row.cweID AS cweID
 MATCH (v:Vulnerability {cveID:cveID})
 MATCH (w:Weakness {cweID:cweID})
 MERGE (v)-[r:EXAMPLE_OF]->(w) // add relationships
 RETURN * ",
 {batchSize: 500}
);

// ###############  import/Vulnerability_affects_product.csv  ####################
// create entity
// label: Product

// uniqueness constraint
CREATE CONSTRAINT UniqueProductName ON (p:Product) ASSERT p.productName IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

//create Product entity and AFFECTS relationship
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///Vulnerability_affects_product.csv')
 YIELD map AS row RETURN row",
 "WITH 
	 row.cveID AS cveID,
	 row.product AS productName, 
	 row.productType AS productType,
	 toInteger(row.nVersions) AS numOfVersion
 MERGE (p:Product {productName:productName}) //add nodes
 ON CREATE SET p.productType=productType
 WITH *
 MATCH (v:Vulnerability {cveID:cveID})
 MERGE (v)-[r:AFFECTS]->(p) // add relationships
 ON CREATE SET r.numOfVersion=numOfVersion
 RETURN * ",
 {batchSize: 500}
);

// add affectedVersion
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///Affects_add_property.csv')
 YIELD map AS row RETURN row",
 "WITH
   row.cveID AS cveID,
row.product AS productName,
   row.version AS version
MATCH (v:Vulnerability{cveID:cveID})-[r:AFFECTS]->(p:Product{productName:productName})
	SET r.affectedVersion = r.affectedVersion + [version]
 RETURN count(*)",
 {batchSize: 500}
);

// ###############  import/Commit_message.csv  ####################
// create entity
// label: Commit

// set uniqueness constraint
CREATE CONSTRAINT UniqueCommitID ON (c:Commit) ASSERT c.commitID IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///Commit_message.csv')
 YIELD map AS row RETURN row",
 "WITH
 row.commitID AS commitID,
 row.commitMessage AS commitMessage
 MERGE (c:Commit {commitID:commitID})
    ON CREATE SET
    c.commitMessage = commitMessage
 RETURN count(*)",
 {batchSize: 500}
);

// create relationship
// relationship type: HAS_COMMIT (Vulnerability HAS_COMMIT Commit)
"CALL apoc.load.csv('file:///Commit_message.csv')
 YIELD map AS row RETURN row",
 "WITH
     row.cveID AS cveID,
     row.commitID AS commitID
 MATCH (v:Vulnerability {cveID:cveID})
 MATCH (c:Commit {commitID:commitID})
 MERGE (v)-[r:HAS_COMMIT]->(c) // add relationships
 RETURN * ",
 {batchSize: 500}
);


##############
MATCH (n:Vulnerability) RETURN count(n); 
MATCH (n:Weakness) RETURN count(n); 
MATCH (n:Product) RETURN count(n); 
MATCH (c:Commit) RETURN count(c)

MATCH p=()-[r:AFFECTS]->() RETURN count(p);
MATCH p=()-[r:EXAMPLE_OF]->() RETURN count(p); 
MATCH p=()-[r:HAS_COMMIT]->() RETURN count(p);


// ############## generate fig 
CALL db.schema.visualization()

