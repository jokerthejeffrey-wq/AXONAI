# Decide when to stop pretrained assistance

The default is `AXON_CHAT_MODE=assisted`. This uses Qwen for general chat while the from-scratch student trains separately. Assistance is not counted as student performance and chats are not automatically training examples.

A falling training loss does not prove that a student can hold a useful conversation. The application deliberately does not switch automatically. Evaluate the published student locally on questions excluded from training and decide whether it meets your project's requirements.

Windows PowerShell (run a separate local server on port 5001):

```powershell
$env:AXON_CHAT_MODE="student"
$env:PORT="5001"
python HTML.py
```

Use http://127.0.0.1:5001/chat to check:

- Several unseen greetings, without repeating the same sentence every turn.
- A two- or three-turn conversation with pronouns and a follow-up question.
- Requests to shorten, clarify or correct an answer.
- Unseen arithmetic and factual questions with verified answers.
- Honest uncertainty instead of confident invented facts.
- A natural response when the user says goodbye or changes the subject.
- Different wording across multiple sampled runs, while keeping facts consistent.

Creator/identity questions use product metadata in both modes, so passing those is NOT evidence that the student learned identity or language. Keep an evaluation log and compare multiple independent examples before promoting a checkpoint. The included 3.42M model and small starter data are not expected to match a pretrained assistant.

To switch the public backend, stop it, set AXON_CHAT_MODE=student in its environment and restart. To return to assistance, use AXON_CHAT_MODE=assisted and restart. The website reads the mode from the API and shows the actual provider. Do not advertise student capability while serving assisted answers.
