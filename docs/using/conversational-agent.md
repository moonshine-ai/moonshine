# Getting Started with a Conversational Agent

- [A Simple Flow](#a-simple-flow)
- [Multi-step Flows](#multi-step-flows)
- [Asking Questions](#asking-questions)
- [Confirmations](#confirmations)
- [Spelled Input](#spelled-input)
- [Error Handling](#error-handling)
- [Agent Setup](#agent-setup)

Many applications need a voice agent that can understand what users are saying and respond appropriately. To make this as straightforward as possible, we let you define different conversational flows. A flow can be as simple as responding to a query, or be a multi-step, branching conversation that takes actions.

## A Simple Flow

To define these flows, you use an [`AgentFlow`](../api/classes.md#agentflow) object, with callbacks that take [`Dialog`](../api/classes.md#dialog) arguments. Here's an example of a simple flow, taken from the [github.com/moonshine-ai/pi-help-bot](https://github.com/moonshine-ai/pi-help-bot) sample code:

```python
    def report_ip_address(d: Dialog):
        ip = _find_local_ip()
        if ip is None:
            yield d.say("Sorry, I couldn't find a local IP address.")
            return
        speech_ip = re.sub(r"(\d)", r"\1 ", ip.replace(".", " dot "))
        yield d.say([
            f"Okay. Your local IP address is {speech_ip}. ",
            f"To repeat, that's {speech_ip}."
        ])

    agent_flow.listen_for("What is my IP address?", report_ip_address)
```

This registers the `report_ip_address()` function to be called whenever the user says anything similar to "What is my IP address?". The matching is done semantically, so alternative phrasings like "Tell me your IP address" or "Can you tell me the local IP address?" should trigger it too. You can register as many top-level conversation starters as you'd like, the system will listen out and route to the closest in meaning.

The function itself receives a `Dialog` argument that represents the current conversational exchange. In this simple case we don't need any additional input from the user so we just use it to `say()` the information that was requested. We break the IP address into separate words for each digit for clarity, and replace the connecting periods with explicit "dot"s, so that 192.178.4.72 becomes "1 9 2 dot 1 7 8 dot 4 dot 72", since that's the conventional way to articulate them in speech.

## Multi-step Flows

For more complex conversations, like setting up a new wifi network, you can define multiple steps and branch points directly in Python:

```python
    def connect_to_wifi(d: Dialog):
        input_ssid = yield d.ask("What's the name of your Wi-Fi network? Say list if you want to pick from a list or spell if you want to spell out the start of the name")
        input_ssid = input_ssid.strip()

        networks = _scan_wifi_networks()

        if input_ssid.lower().strip(string.punctuation) == "list":
            yield d.say("Say yes to the network you want to connect to.")
            for network in networks:
                if (yield d.confirm(f"{network}?")):
                    input_ssid = network
                    break
        elif input_ssid.lower().strip(string.punctuation) == "spell":
            input_ssid = yield d.ask("Spell out the start of the network name.", mode=SPELLED)

        found_ssid = fuzzy_match_network(input_ssid, networks)
        if found_ssid is None:
            yield d.say(f"Sorry, I couldn't find a matching network for {input_ssid}.")
            return

        password = yield d.ask(
            f"Please spell the Wi-Fi password for {found_ssid} one character at a time, and say done when finished.",
            mode=SPELLED,
        )

        yield d.say(f"Connecting to {found_ssid}.")

        try:
            result = subprocess.run(
                ["sudo", "nmcli", "device", "wifi",
                    "connect", found_ssid, "password", password],
                capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError:
            yield d.say("Sorry, network manager was not found on this system.")
            return
        except subprocess.TimeoutExpired:
            yield d.say("Sorry, the connection attempt timed out.")
            return

        if result.returncode == 0:
            yield d.say(f"Connected to {found_ssid}.")
        else:
            print(f"[ERROR] nmcli stderr: {result.stderr}", file=sys.stderr)
            yield d.say(
                f"Sorry, I wasn't able to connect to {found_ssid}. "
                "Please check the network name and password and try again."
            )

    agent_flow.listen_for("Connect to Wi-Fi", connect_to_wifi)
```

## Asking Questions

The first thing the function does is ask the user to give them the name of the network they want to join, through the call:

```python
input_ssid = yield d.ask("What's the name of your Wi-Fi network?...")
```

The Dialog class lets you ask users questions and will return the string containing the what they said in response. The only unusual feature here, compared to regular Python code, is the `yield` keyword. Because it may take some time for the user to respond, we call yield to hand back control to the main script until their response has been received. This is a general pattern for `AgentFlow` and you'll see it wherever we're waiting for the user to say something, to avoid blocking.

## Confirmations

```python
        if input_ssid.lower().strip(string.punctuation) == "list":
            yield d.say("Say yes to the network you want to connect to.")
            for network in networks:
                if (yield d.confirm(f"{network}?")):
                    input_ssid = network
                    break
```

Our example application supports a few different input methods - running through a list of networks, spelling out the first few letters, or saying the name. Here we implement the list approach by looping through all the available networks and asking the user whether each is the one they want. Here you can see that regular loops and conditional statements work as you'd expect in Python.

For each network, we call `confirm()`, which asks a question and then waits for a positive or negative result. Like all matching in the system this is done semantically, so "okay", "affirmative", and "go ahead" will work as well as a straightforward "yes".

## Spelled Input

```python
        password = yield d.ask(
            f"Please spell the Wi-Fi password for {found_ssid} one character at a time, and say done when finished.",
            mode=SPELLED,
        )
```

Password input is tricky, because they consist of arbitrary letters, digits, and symbols, and so they have to be spelled out by the user. Moonshine supports this through the `mode=SPELLED` argument. This asks the user to spell out each character, and uses a fine-tuned model to recognise what the user is saying for each. As well as supporting regular utterances like "aitch" or "capital zee", it also supports the NATO alphabet ("alpha", "bravo", etc) and even short descriptive phrases like "E as in elephant". It repeats back what it heard, and lets you delete mistakes.

## Error Handling

```python
        try:
            result = subprocess.run(
                ["sudo", "nmcli", "device", "wifi",
                    "connect", found_ssid, "password", password],
                capture_output=True, text=True, timeout=30,
            )
        except FileNotFoundError:
            yield d.say("Sorry, network manager was not found on this system.")
            return
        except subprocess.TimeoutExpired:
            yield d.say("Sorry, the connection attempt timed out.")
            return
```

The flow also works with other control structures like exception handlers, so you can specify your conversations using idiomatic code, even for error recovery.

## Agent Setup

Once your flows are written, the only setup left is to register them and go live. `AgentFlow` opens everything it needs itself — the speech recognition model, the microphone, and the speech synthesizer — so there's nothing to wire together:

```python
    agent_flow = (
        AgentFlow()
        .language("en")
        .listen_for("What is my IP address?", report_ip_address)
        .listen_for("Connect to Wi-Fi", connect_to_wifi)
    )

    agent_flow.start_listening()
```

Every configuration method returns the runner, so a whole voice interface can be built in a single expression, and each one has a working default. `listen_for()` registers the conversation starters. "Cancel" and "start over" need no registration: they work at any point inside a flow, and outside one they're treated as ordinary speech so a dictation interface doesn't lose them. Use `always()` to register a phrase of your own that stays live at every moment, whether or not a flow is running.

`start_listening()` opens and downloads whatever is missing on first use (the embedding model used for matching, the speech to text model for your language, and a synthesizer), then returns as soon as the microphone is live. Speech arrives on the audio thread and drives your flows from there, so your own code is free to sleep, run a UI, or do anything else. If you want the loading to happen at a moment of your choosing rather than on the first `start_listening()` call, call `load()` yourself beforehand and pass `on_progress()` a callback to report download progress. Call `close()` when you're finished to release everything the runner opened.

To give this a try for yourself, run this built-in example:

<!-- doc-test: parse-only -->
```bash
python -m moonshine_voice.agent_flow
```
