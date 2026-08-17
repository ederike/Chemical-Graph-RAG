from pathlib import Path
from .database import BaseDB,BaseVDB

class DocDB(BaseDB):
    def __init__(self,db_path):
        name='doc'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT DEFAULT 'new',
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone', 
                hash TEXT UNIQUE
            );
            """
        super().__init__(db_path,name,create_table_sql)

class ChunkDB(BaseDB):
    def __init__(self,db_path):
        name='chunk'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT DEFAULT 'new',
                doc_id INTEGER,
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone'
            );
            """
        super().__init__(db_path,name,create_table_sql)

class HyperedgeDB(BaseDB):
    def __init__(self,db_path):
        name='hyperedge'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT,
                doc_id INTEGER,
                chunk_id INTEGER,
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone'
            );
            """
        super().__init__(db_path,name,create_table_sql)

class NodeDB(BaseDB):
    def __init__(self,db_path):
        name='node'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT,
                doc_id INTEGER,
                chunk_id INTEGER,
                hyperedge_id INTEGER,
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone'
            );
            """
        super().__init__(db_path,name,create_table_sql)

class EdgeDB(BaseDB):
    def __init__(self,db_path):
        name='edge'
        create_table_sql = \
            f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                status TEXT,
                doc_id INTEGER,
                chunk_id INTEGER,
                hyperedge_id INTEGER,
                src_node_id INTEGER,
                dst_node_id INTEGER,
                name TEXT,
                content TEXT,
                extra TEXT,
                tokens INT,
                embedding_content TEXT,
                embedding_status TEXT DEFAULT 'undone'
            );
            """
        super().__init__(db_path,name,create_table_sql)

class DocVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'doc', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )

class ChunkVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'chunk', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )

class HyperedgeVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'hyperedge', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )

class NodeVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'node', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )

class EdgeVDB(BaseVDB):
    def __init__(self, vdb_path, dim, shard_max_vectors=None, **index_kwargs):
        super().__init__(
            vdb_path, 'edge', dim,
            shard_max_vectors=shard_max_vectors, **index_kwargs,
        )
