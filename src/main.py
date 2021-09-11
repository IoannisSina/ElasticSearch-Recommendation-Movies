from elasticsearch import Elasticsearch
es = Elasticsearch()

es.indices.create(index="first_index")
print(es.indices.exists(index="first_index"))






"""
{
  "name" : "DESKTOP-0QIB0R2",
  "cluster_name" : "elasticsearch",
  "cluster_uuid" : "VPPzbUQ4RZSXLXJevdj3ig",
  "version" : {
    "number" : "7.14.1",
    "build_flavor" : "default",
    "build_type" : "zip",
    "build_hash" : "66b55ebfa59c92c15db3f69a335d500018b3331e",
    "build_date" : "2021-08-26T09:01:05.390870785Z",
    "build_snapshot" : false,
    "lucene_version" : "8.9.0",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  },
  "tagline" : "You Know, for Search"
}
"""