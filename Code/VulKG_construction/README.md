# Vulnerability Knowledge Graph Construction

Folder **import** contains all original data for VulKG deployment.

File **VulKG_Deployment.cypher** contains the cypher codes for VulKG deployment on the Neo4j graph database platform. 


## VulKG Deployment
This section introduces how to deploy the VulKG into the Neo4j graph database platform.
### Programming Language
[Cypher Query Language](https://neo4j.com/developer/cypher/)
### Software/Platform
Neo4j Desktop
### A step-by-step guide for VulKG Deployment
1. Download [(from here)](https://neo4j.com/download/) and install [(refer here)]( https://neo4j.com/docs/desktop-manual/current/installation/) Neo4j Desktop 1.6.1 or higher versions
2. Create a project named **VulKG Project** with Neo4j Desktop.
3. Add a local DBMS named **Graph DBMS** with Neo4j Desktop and set the password as **Neo4j**. Choose version 4.4.11 for **Graph DBMS**.
4. Start **Graph DBMS**
5. Install APOC [(refer here)](https://neo4j.com/labs/apoc/4.3/installation/) and Graph Data Science Library [(refer here)](https://neo4j.com/docs/graph-data-science/current/installation/neo4j-desktop/) plugins for **Graph DBMS** with Neo4j Desktop.
 
6. Open the setting file of **Graph DBMS** and add a line as below in the setting file. 
```
    apoc.import.file.enabled=true
```
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;to tackle an error:

```
    Failed to invoke procedure `apoc.periodic.iterate`: Caused by: 
    java.lang.RuntimeException: Import from files not enabled, please set 
    apoc.import.file.enabled=true in your apoc.conf
```
7. Put all files in the [import](import/) folder into the **import** folder of **Graph DBMS**.
8. Open **Graph DBMS** with **Neo4j Browser**. Since Neo4j Browser comes out-of-the-box when you install Neo4j Desktop on your system, no installation is required.
9.  Click the **Enable multi statement query editor** to enable running multiple Cypher statements separated by semi-colons **;** in the Neo4j Browser setting.
10. Run Cypher statements in the **2.VulKG_Deployment_Cypher.cypher** file with the Neo4j Browser to deploy VulKG.

