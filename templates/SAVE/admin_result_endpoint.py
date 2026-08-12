"""
Admin Result URL Endpoint
Add this endpoint to your app.py or admin routes file.

This returns the result page URL with video and data URLs pre-loaded,
identical to how the email result link works.
"""

# Add this endpoint to your Flask app:

@app.get("/api/admin/result-url/<job_id>")
def admin_result_url(job_id):
    """Get result page URL for a completed job (same as email link)."""
    import os
    import json
    from urllib.parse import urlencode

    token = request.args.get("token")
    if token != os.environ.get("ADMIN_TOKEN"):
        return jsonify({"error": "Unauthorized"}), 401

    # Get the job from database
    job = UserJob.query.filter_by(job_id=job_id).first()
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job.status != 'complete':
        return jsonify({"error": f"Job not complete (status: {job.status})"}), 400

    # Get R2 public URL base
    r2_public_url = os.environ.get("R2_PUBLIC_URL", "")
    if not r2_public_url:
        return jsonify({"error": "R2_PUBLIC_URL not configured"}), 500

    # Build video URL (assumes object_key is stored in job)
    video_url = f"{r2_public_url}/{job.object_key}"

    # Build data URL - results JSON should be stored in R2
    # Option 1: If results_json_key is stored in the job
    if hasattr(job, 'results_json_key') and job.results_json_key:
        data_url = f"{r2_public_url}/{job.results_json_key}"
    # Option 2: Generate consistent path based on job_id
    else:
        data_url = f"{r2_public_url}/results/{job_id}.json"

    # Build result page URL with query parameters
    base_url = os.environ.get("BASE_URL", request.host_url.rstrip('/'))
    params = urlencode({
        'video': video_url,
        'data': data_url
    })
    result_url = f"{base_url}/result?{params}"

    return jsonify({
        "url": result_url,
        "video_url": video_url,
        "data_url": data_url
    })


# =============================================================
# Alternative: If you need to upload results JSON to R2 first
# =============================================================

def upload_results_to_r2(job_id, results_data):
    """
    Upload results JSON to R2 and return the object key.
    Call this when job completes, before sending email.
    """
    import json
    import boto3
    import os

    s3 = boto3.client(
        's3',
        endpoint_url=os.environ.get('R2_ENDPOINT'),
        aws_access_key_id=os.environ.get('R2_ACCESS_KEY'),
        aws_secret_access_key=os.environ.get('R2_SECRET_KEY'),
        region_name='auto'
    )

    object_key = f"results/{job_id}.json"

    s3.put_object(
        Bucket=os.environ.get('R2_BUCKET'),
        Key=object_key,
        Body=json.dumps(results_data),
        ContentType='application/json'
    )

    return object_key


# =============================================================
# Updated jobs endpoint to include status
# =============================================================

# Make sure your /api/admin/jobs endpoint returns the status field:

@app.get("/api/admin/jobs")
def admin_jobs():
    """Get all jobs with status and file size."""
    token = request.args.get("token")
    if token != os.environ.get("ADMIN_TOKEN"):
        return jsonify({"error": "Unauthorized"}), 401

    jobs = UserJob.query.order_by(UserJob.created_at.desc()).all()

    return jsonify({
        "total": len(jobs),
        "jobs": [{
            "job_id": j.job_id,
            "user_email": j.user.email if j.user else None,
            "filename": j.filename,
            "file_size": j.file_size,  # <-- File size in bytes
            "object_key": j.r2_object_key,
            "status": j.status,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "completed_at": j.completed_at.isoformat() if hasattr(j, 'completed_at') and j.completed_at else None
        } for j in jobs]
    })
