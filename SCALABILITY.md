# Production scalability notes

This application includes defensive limits, caching, guarded parsing and expensive-analysis caps.

## Important

A Streamlit application is not, by itself, a 1,000-concurrent-user production architecture.
For 1,000+ simultaneous users, deploy multiple Streamlit replicas behind a load balancer,
with appropriate CPU/RAM capacity and observability. Shared persistent storage/database
should be used for durable results rather than process-local session state.

Recommended production architecture:

Browser → Load Balancer → multiple Streamlit replicas
                         ↘ shared object/database storage for persistent jobs/results

For very large FASTA jobs, move alignment/phylogeny into background workers and return
job status/results to the UI. Do not promise zero errors under arbitrary load; capacity
testing and autoscaling are required.
