read_file = {
    "type" : "function",
    "function" : {
        "name" : "read_file",
        "description" : "Read the contents of a file at the given path.",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "path" : {
                    "type" : "string",
                    "description" : "The path to the file to read."
                }
            },
            "required" : ["path"]
        }
    }
}

write_file = {
    "type" : "function",
    "function" : {
        "name" : "write_file",
        "description" : "Write the given content to a file at the specified path.",
        "parameters" : {
            "type" : "object",
            "properties" : {
                "path" : {
                    "type" : "string",
                    "description" : "The path to the file to write."
                },
                "content" : {
                    "type" : "string",
                    "description" : "The content to write to the file."
                }
            },
            "required" : ["path", "content"]
        }
    }
}

TOOLS = [read_file, write_file]
TOOLS_HANDLER = {
    "read_file": read_file,
    "write_file": write_file,
}
