import os
if os.environ.get('SONAR_HOST_STAGED_ALLREDUCE') == '1':
    try:
        import adapter
        adapter.install()
    except Exception:
        import traceback
        traceback.print_exc()
        os._exit(78)
