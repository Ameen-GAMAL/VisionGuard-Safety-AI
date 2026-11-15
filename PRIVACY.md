# Privacy and evidence handling

VisionGuard is a research prototype for human-reviewed workplace safety analytics. It cannot certify that a site is safe. Missed detections, false positives, incorrect distance calibration, and identity mix-ups are possible. Keep established monitoring and emergency procedures in place.

## Data flow

| Information | Processing and storage |
| --- | --- |
| Input video or camera frames | The local Python process reads frames. A browser connected to the Streamlit server can receive rendered media and download reports. Run the demo on loopback with authorized footage. |
| Annotated clips, screenshots, heatmaps and PDFs | The configured output folders retain these files. Set `events.save_screenshots: false` for new runs that should skip separate incident stills. Annotated video and heatmaps can still contain identifying or sensitive site information. Review all media and reports before sharing. |
| Event history and logs | The local SQLite database stores run source names, relative incident times, run-local IDs, descriptions and statistics. Application logs may include source paths. The project does not provide field-level redaction or application-level encryption. |
| Re-identification | The opt-in CLI matches appearance embeddings across camera sequences and writes visual montages. Treat embeddings, tracks and crops as potentially identifying. A matching score cannot establish a person's identity. |
| Assistant | Facts and optional embeddings are processed locally by default. Initial model acquisition can contact a model registry. With `assistant.cloud_enabled: true`, an environment API key and affirmative user consent, the question, aggregate facts and relevant incident text with relative timestamps are sent to Anthropic. The provider request omits image/video bytes, source paths, and the run’s calendar date. Text can still contain zone labels, run-local track IDs, or sensitive details a user enters. |

## Handling a real trial

1. Obtain authorization for the footage and document the purpose and operator. Have the organization assess required notice, legal basis and any consent obligations before collecting or processing footage. Keep Re-ID and cloud synthesis off unless reviewed separately.
2. Choose the minimum fields, duration and copies needed for the purpose. Keep outputs on a restricted, encrypted volume. Use operating-system access controls, restrict dashboard access and use network isolation. The repository’s local Streamlit settings bind loopback and switch off usage telemetry.
3. Set a written retention period before the trial. VisionGuard leaves files and SQLite runs in place until the operator removes them. After expiration, stop the app and remove the approved source copies, corresponding database records or dedicated trial database, annotated clips, report files, screenshots, Re-ID montages and logs. Include backups and exported copies in the retention plan. Separate trial storage simplifies confirming deletion.
4. Use only rights-cleared stock or generated examples in public evidence. Redact faces, names, screens, site identifiers and embedded metadata as appropriate. A stock license or permission to process footage does not establish permission to make a public claim about a person shown in it.
5. Review applicable [Anthropic terms and data handling](https://privacy.anthropic.com/) before enabling cloud synthesis. Use a secret manager or process environment for the key. Do not include keys, private labels, or applicant or employee details in prompts, logs, CI artifacts or issue reports.

The externally linked Space has its own operator, data handling and source revision. See [the README](README.md) for the status and use the local run path for controlled testing.
