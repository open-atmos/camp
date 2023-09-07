# Define the repository URL and destination path
repo_url="git@github.com:mattldawson/cvode.git" # Use the SSH URL
destination_path="../../../cvode-3.4-alpha"

# Check if the destination directory already exists
if [ -d "$destination_path" ]; then
    echo "Destination directory already exists. Skipping cloning."
else
    # Clone the repository using SSH
    git clone "$repo_url" "$destination_path"
    if [ $? -eq 0 ]; then
        echo "Repository cloned successfully to: $destination_path"
    else
        echo "Failed to clone the repository."
    fi
fi
