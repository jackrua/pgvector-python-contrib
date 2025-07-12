# Use the official PostgreSQL base image
FROM postgres:16

# Set environment variables for default PostgreSQL user and database
ENV POSTGRES_USER=myuser
ENV POSTGRES_PASSWORD=mypassword
ENV POSTGRES_DB=mydatabase

# Optional: Copy initialization SQL scripts (if you have any)
# These scripts will be run when the container is initialized
# COPY ./init.sql /docker-entrypoint-initdb.d/

# Expose the PostgreSQL port
EXPOSE 5432
