package ai.moonshine.voice;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import android.content.Context;

import androidx.test.InstrumentationRegistry;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Exercises the conversation runner without any models: prompts are captured
 * through {@code speakWith} and answers are fed in through
 * {@link AgentFlow#handleUtterance}, so trigger matching falls back to the
 * substring path and nothing is downloaded.
 */
public class AgentFlowTest {

    private Context context;
    private final List<String> spoken = new CopyOnWriteArrayList<>();

    @Before
    public void setUp() {
        context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        spoken.clear();
    }

    private AgentFlow newAgent() {
        return new AgentFlow(context).microphone(false).speakWith(spoken::add);
    }

    /**
     * Waits for the flow thread to speak its {@code promptCount}-th prompt and
     * park on the answer, then feeds {@code answer} in.
     */
    private void answer(AgentFlow agent, int promptCount, String answer) {
        long deadline = System.currentTimeMillis() + 2000;
        while (spoken.size() < promptCount || !agent.isAwaitingAnswer()) {
            if (System.currentTimeMillis() > deadline) {
                throw new AssertionError("Timed out waiting for prompt " + promptCount
                        + "; heard " + spoken);
            }
            sleep();
        }
        agent.handleUtterance(answer);
    }

    private static void sleep() {
        try {
            Thread.sleep(5);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Waits for a main-thread callback to have delivered {@code count} items. */
    private static void awaitCount(List<String> items, int count) {
        long deadline = System.currentTimeMillis() + 2000;
        while (items.size() < count) {
            if (System.currentTimeMillis() > deadline) {
                throw new AssertionError(
                        "Timed out waiting for " + count + " item(s); have " + items);
            }
            sleep();
        }
    }

    @Test
    public void testRunsAFlowToCompletion() {
        AgentFlow agent = newAgent();
        AtomicReference<String> captured = new AtomicReference<>();
        agent.listenFor("set up wifi", d -> {
            String ssid = d.ask("What's the name of your wifi network?");
            captured.set(ssid);
            if (d.confirm("I heard " + ssid + ". Is that right?")) {
                d.say("Done. Connecting to " + ssid + ".");
            }
        });

        agent.handleUtterance("set up wifi");
        answer(agent, 1, "home network");
        answer(agent, 2, "yes");
        assertTrue(agent.waitUntilIdle(3000));

        assertEquals("home network", captured.get());
        assertEquals(Arrays.asList(
                "What's the name of your wifi network?",
                "I heard home network. Is that right?",
                "Done. Connecting to home network."), new ArrayList<>(spoken));
        agent.close();
    }

    @Test
    public void testCancelStopsTheFlow() {
        AgentFlow agent = newAgent();
        AtomicBoolean finished = new AtomicBoolean(false);
        agent.listenFor("set up wifi", d -> {
            d.ask("What's the name of your wifi network?");
            finished.set(true);
        });

        agent.handleUtterance("set up wifi");
        answer(agent, 1, "cancel");
        assertTrue(agent.waitUntilIdle(3000));

        assertFalse("cancel should unwind the flow before it finishes", finished.get());
        assertFalse(agent.isActive());
        agent.close();
    }

    @Test
    public void testStartOverRunsTheFlowAgain() {
        AgentFlow agent = newAgent();
        AtomicBoolean restarted = new AtomicBoolean(false);
        agent.listenFor("set up wifi", d -> {
            if (restarted.get()) {
                d.say("Second time around.");
                return;
            }
            restarted.set(true);
            d.ask("What's the name of your wifi network?");
        });

        agent.handleUtterance("set up wifi");
        answer(agent, 1, "start over");
        assertTrue(agent.waitUntilIdle(3000));

        assertTrue(spoken.contains("Second time around."));
        agent.close();
    }

    @Test
    public void testBuiltInGlobalsDoNotClaimSpeechOutsideAFlow() {
        AgentFlow agent = newAgent();
        List<String> leftovers = new CopyOnWriteArrayList<>();
        agent.otherwise(leftovers::add);
        agent.listenFor("set up wifi", d -> d.ask("Name?"));

        // Nothing is active, so there is no flow for either phrase to act on and they are just
        // words. Claiming them here would lose a line of dictation.
        agent.handleUtterance("cancel");
        agent.handleUtterance("start over");
        agent.handleUtterance("cancel my subscription tomorrow");
        awaitCount(leftovers, 3);

        assertEquals(
                Arrays.asList("cancel", "start over", "cancel my subscription tomorrow"),
                new ArrayList<>(leftovers));
        agent.close();
    }

    @Test
    public void testRegisteringABuiltInPhraseWithAlwaysMakesItLive() {
        AgentFlow agent = newAgent();
        List<String> leftovers = new CopyOnWriteArrayList<>();
        List<String> cancels = new CopyOnWriteArrayList<>();
        agent.otherwise(leftovers::add);
        agent.always("cancel", d -> cancels.add("cancel"));

        agent.handleUtterance("cancel");
        awaitCount(cancels, 1);

        assertEquals(Arrays.asList("cancel"), new ArrayList<>(cancels));
        assertTrue(leftovers.isEmpty());
        agent.close();
    }

    @Test
    public void testRepromptsWhenTheAnswerMakesNoSense() {
        AgentFlow agent = newAgent();
        AtomicReference<String> picked = new AtomicReference<>();
        Map<String, List<String>> choices = new LinkedHashMap<>();
        choices.put("wifi", Arrays.asList("wireless"));
        choices.put("bluetooth", Arrays.asList("bt"));
        agent.listenFor("connect", d -> picked.set(d.choose("Which one?", choices)));

        agent.handleUtterance("connect");
        answer(agent, 1, "something else entirely");
        answer(agent, 2, "bluetooth");
        assertTrue(agent.waitUntilIdle(3000));

        assertEquals("bluetooth", picked.get());
        assertEquals("expected one prompt plus one re-prompt", 2, spoken.size());
        assertTrue(spoken.get(1).contains("Which one?"));
        agent.close();
    }

    @Test
    public void testOtherwiseSeesOnlyUnclaimedLines() {
        AgentFlow agent = newAgent();
        List<String> leftovers = new CopyOnWriteArrayList<>();
        agent.otherwise(leftovers::add);
        agent.listenFor("start setup", d -> d.ask("Name?"));

        agent.handleUtterance("the weather is nice today");
        awaitCount(leftovers, 1);
        assertEquals(Arrays.asList("the weather is nice today"), new ArrayList<>(leftovers));

        // A trigger phrase belongs to the flow it starts, and the answer that
        // follows belongs to the prompt waiting for it.
        agent.handleUtterance("start setup");
        answer(agent, 1, "Alice");
        assertTrue(agent.waitUntilIdle(3000));

        assertEquals(1, leftovers.size());
        assertTrue(spoken.contains("Name?"));
        agent.close();
    }

    @Test
    public void testSilencedRunnerStillAdvancesFlows() {
        // No speakWith here: speech(false) is the whole configuration, and the prompts have to
        // keep flowing without a synthesizer behind them.
        AgentFlow agent = new AgentFlow(context).microphone(false).speech(false);
        List<String> said = new CopyOnWriteArrayList<>();
        agent.onSaid(said::add);
        agent.listenFor("begin", d -> {
            String name = d.ask("Name?");
            d.say("Hello, " + name + ".");
        });

        agent.handleUtterance("begin");
        awaitCount(said, 1);
        long deadline = System.currentTimeMillis() + 2000;
        while (!agent.isAwaitingAnswer() && System.currentTimeMillis() < deadline) {
            sleep();
        }
        agent.handleUtterance("Alice");
        assertTrue(agent.waitUntilIdle(3000));
        awaitCount(said, 2);

        assertEquals(Arrays.asList("Name?", "Hello, Alice."), new ArrayList<>(said));
        agent.close();
    }

    @Test
    public void testReportsFlowErrors() {
        AgentFlow agent = newAgent();
        AtomicReference<Throwable> reported = new AtomicReference<>();
        agent.onError(reported::set);
        agent.listenFor("break things", d -> {
            throw new IllegalStateException("boom");
        });

        agent.handleUtterance("break things");
        assertTrue(agent.waitUntilIdle(3000));

        long deadline = System.currentTimeMillis() + 2000;
        while (reported.get() == null && System.currentTimeMillis() < deadline) {
            sleep();
        }
        assertTrue(reported.get() instanceof IllegalStateException);
        agent.close();
    }
}
